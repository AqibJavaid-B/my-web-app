import os
import io
import json
import logging
import time
import uuid
from datetime import datetime
from urllib.parse import unquote_plus

from flask import Flask, request, render_template_string
import boto3
from botocore.config import Config
from botocore.exceptions import UnknownServiceError
from PyPDF2 import PdfReader

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Config / env ===
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-oss-120b-1:0",
)
BEDROCK_TIMEOUT_CONFIG = Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 2})

S3_BUCKET = os.environ.get("S3_BUCKET") or os.environ.get("UPLOAD_BUCKET") or os.environ.get("S3_BUCKET_NAME")
DDB_TABLE = os.environ.get("DDB_TABLE") or os.environ.get("DDB_TABLE_NAME")
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
MIN_CONTEXT_CHARS = int(os.environ.get("MIN_CONTEXT_CHARS", "3000"))
EXCERPT_SIZE = int(os.environ.get("EXCERPT_SIZE", "3200"))

# === boto3 clients ===
s3 = boto3.client("s3", region_name=AWS_REGION) if AWS_REGION else boto3.client("s3")
dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION) if AWS_REGION else boto3.resource("dynamodb")

# === Lazy Bedrock client ===
_bedrock_client = None
_bedrock_available = None


def get_bedrock_client():
    global _bedrock_client, _bedrock_available
    if _bedrock_available is False:
        return None
    if _bedrock_client is not None:
        return _bedrock_client
    try:
        _bedrock_client = boto3.client("bedrock-runtime", config=BEDROCK_TIMEOUT_CONFIG, region_name=AWS_REGION)
        _bedrock_available = True
        app.logger.info("Bedrock client initialized.")
        return _bedrock_client
    except UnknownServiceError:
        _bedrock_available = False
        app.logger.error(
            "Bedrock client not available: botocore/boto3 in image does not include 'bedrock-runtime'. Upgrade boto3/botocore."
        )
        return None
    except Exception as e:
        _bedrock_available = False
        app.logger.exception("Failed to initialize Bedrock client: %s", e)
        return None


def call_bedrock(prompt, max_retries=2, backoff_sec=0.5):
    """
    Calls Bedrock runtime using lazy client. Returns the text response.
    Raises RuntimeError if client not available.
    """
    client = get_bedrock_client()
    if client is None:
        raise RuntimeError("Bedrock client not available. Check boto3/botocore version and container network/region.")

    payload = {"text": prompt}
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(payload).encode("utf-8"),
            )
            body_bytes = resp.get("body").read()
            text = None
            try:
                j = json.loads(body_bytes.decode("utf-8"))
                if isinstance(j, dict):
                    if "results" in j and isinstance(j["results"], list) and j["results"]:
                        candidate = j["results"][0]
                        if isinstance(candidate, dict) and "outputText" in candidate:
                            text = candidate["outputText"]
                    if not text and "content" in j:
                        text = j["content"]
                    if not text and "output" in j:
                        text = j["output"]
                if text is None:
                    text = json.dumps(j)
            except Exception:
                try:
                    text = body_bytes.decode("utf-8", errors="ignore")
                except Exception:
                    text = str(body_bytes)
            return text.strip()
        except Exception as e:
            last_exc = e
            app.logger.warning("Bedrock invoke_model attempt %d failed: %s", attempt, e)
            time.sleep(backoff_sec * (2 ** (attempt - 1)))

    app.logger.exception("Bedrock invocation failed after %d attempts. Last error: %s", max_retries, last_exc)
    raise last_exc


# === Helpers: PDF/text extraction, S3, DynamoDB ===
def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract text from PDF binary using PyPDF2.
    Returns a string (may be empty).
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text:
                texts.append(text)
        return "\n".join(texts)
    except Exception as e:
        app.logger.warning("PDF extraction failed: %s", e)
        return ""


def save_doc_to_s3_and_dynamo(file_obj, filename):
    """
    Save uploaded file bytes to S3 and metadata to DynamoDB.
    Returns doc_id.
    """
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET env var not set")
    if not DDB_TABLE:
        raise RuntimeError("DDB_TABLE env var not set")

    doc_id = uuid.uuid4().hex[:8]
    s3_key = f"uploads/{doc_id}/{filename}"
    content = file_obj.read()

    # Upload original file
    s3.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=content)

    excerpt = ""
    excerpt_key = ""
    if filename.lower().endswith(".pdf"):
        excerpt = extract_text_from_pdf_bytes(content)[:EXCERPT_SIZE]
        if excerpt:
            excerpt_key = f"uploads/{doc_id}/{filename}.excerpt.txt"
            s3.put_object(Bucket=S3_BUCKET, Key=excerpt_key, Body=excerpt.encode("utf-8"))
    else:
        # attempt to decode as text
        try:
            text = content.decode("utf-8")
        except Exception:
            try:
                text = content.decode("latin-1", errors="ignore")
            except Exception:
                text = ""
        excerpt = text[:EXCERPT_SIZE]
        if excerpt:
            excerpt_key = f"uploads/{doc_id}/{filename}.excerpt.txt"
            s3.put_object(Bucket=S3_BUCKET, Key=excerpt_key, Body=excerpt.encode("utf-8"))

    # Save metadata
    table = dynamodb.Table(DDB_TABLE)
    item = {
        "doc_id": doc_id,
        "s3_key": s3_key,
        "filename": filename,
        "excerpt_s3_key": excerpt_key or "",
        "uploaded_at": int(time.time()),
    }
    table.put_item(Item=item)
    return doc_id


def get_doc_excerpt(doc_id):
    """
    Retrieve excerpt text for a given doc_id from DynamoDB + S3.
    Returns excerpt string or None.
    """
    if not DDB_TABLE:
        return None
    table = dynamodb.Table(DDB_TABLE)
    try:
        resp = table.get_item(Key={"doc_id": doc_id})
    except Exception as e:
        app.logger.exception("DynamoDB get_item failed: %s", e)
        return None
    item = resp.get("Item")
    if not item:
        return None
    excerpt_key = item.get("excerpt_s3_key") or ""
    if excerpt_key:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=excerpt_key)
            return obj["Body"].read().decode("utf-8", errors="ignore")
        except Exception as e:
            app.logger.warning("Failed to read excerpt from S3: %s", e)
            return None
    return None


def list_recent_docs(limit=50):
    """
    Return recent documents (scan). For simple MVP only.
    """
    if not DDB_TABLE:
        return []
    table = dynamodb.Table(DDB_TABLE)
    try:
        resp = table.scan(Limit=limit)
        items = resp.get("Items", []) or []
        items_sorted = sorted(items, key=lambda x: x.get("uploaded_at", 0), reverse=True)
        return items_sorted
    except Exception as e:
        app.logger.exception("DynamoDB scan failed: %s", e)
        return []


# === Conversation save (optional) ===
def save_conversation(doc_id, question, answer):
    if not DDB_TABLE:
        return
    try:
        ddb = boto3.resource("dynamodb", region_name=AWS_REGION) if AWS_REGION else boto3.resource("dynamodb")
        table = ddb.Table(DDB_TABLE)
        item = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "question": question,
            "answer": answer,
            "ts": int(datetime.utcnow().timestamp()),
        }
        table.put_item(Item=item)
    except Exception:
        logger.exception("Failed to save conversation to DynamoDB")


# === Web UI templates ===
INDEX_HTML = """
<!doctype html>
<title>DocChat</title>
<h1>DocChat — Ask questions about your document</h1>

<form action="/upload" method=post enctype=multipart/form-data>
  <label>Upload plain text file (.txt) or PDF (.pdf):</label>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>

<form action="/ask" method=post>
  <label>Document ID (from upload):</label>
  <input type=text name=doc_id placeholder="document id">
  <br>
  <label>Question:</label>
  <input type=text name=question style="width:60%">
  <input type=submit value="Ask">
</form>

<h2>Uploaded documents (recent)</h2>
<ul>
{% for doc in docs %}
  <li><strong>{{doc.doc_id}}</strong> — {{doc.filename}} — uploaded_at: {{doc.uploaded_at}}</li>
{% endfor %}
</ul>
"""


# === Flask routes ===
@app.route("/", methods=["GET"])
def index():
    docs = list_recent_docs(limit=50)
    return render_template_string(INDEX_HTML, docs=docs)


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return "No file uploaded", 400
    filename = f.filename or f"upload-{int(time.time())}"
    try:
        doc_id = save_doc_to_s3_and_dynamo(f, filename)
    except Exception as e:
        app.logger.exception("Upload save failed: %s", e)
        return "Upload failed", 500
    return f"Uploaded as document id: {doc_id}. Go back and ask a question. <a href='/'>Home</a>"


@app.route("/ask", methods=["POST"])
def ask():
    doc_id = (request.form.get("doc_id") or "").strip()
    question = (request.form.get("question") or "").strip()
    if not doc_id:
        return "Missing doc_id", 400
    if not question:
        return "Question empty", 400

    excerpt = get_doc_excerpt(doc_id)
    if not excerpt:
        return "Document not found or no text extracted", 404

    prompt = (
        "You are a helpful assistant that answers questions about the provided document.\n\n"
        f"Document (excerpt):\n{excerpt[:MIN_CONTEXT_CHARS]}\n\n"
        "Answer the question below concisely. If the answer is not present in the document, say \"I don't know based on the document.\" Do not hallucinate.\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    try:
        answer = call_bedrock(prompt)
    except Exception as e:
        app.logger.exception("Bedrock error: %s", e)
        return f"Bedrock error: {e}", 500

    # optionally save conversation
    try:
        save_conversation(doc_id, question, answer)
    except Exception:
        pass

    return render_template_string("<h3>Answer</h3><pre>{{answer}}</pre><p><a href='/'>Back</a></p>", answer=answer)


if __name__ == "__main__":
    # for local dev: set PORT env var or default to 80
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
