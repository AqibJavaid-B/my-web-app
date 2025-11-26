import os
import json
import logging
from flask import Flask, request, render_template_string, jsonify, redirect, url_for
import boto3
from botocore.config import Config
from botocore.exceptions import UnknownServiceError
from urllib.parse import unquote_plus
from datetime import datetime
import uuid

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bedrock client
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-oss-120b-1:0")
BEDROCK_TIMEOUT_CONFIG = Config(read_timeout=60, connect_timeout=5, retries={"max_attempts": 2})
# bedrock = boto3.client("bedrock-runtime", config=BEDROCK_TIMEOUT_CONFIG)

# Lazy-initialized bedrock client
_bedrock_client = None
_bedrock_available = None

def get_bedrock_client():
    global _bedrock_client, _bedrock_available
    if _bedrock_available is False:
        return None
    if _bedrock_client is not None:
        return _bedrock_client
    try:
        _bedrock_client = boto3.client("bedrock-runtime", config=BEDROCK_TIMEOUT_CONFIG)
        _bedrock_available = True
        return _bedrock_client
    except UnknownServiceError:
        # SDK doesn't know about bedrock-runtime (old botocore)
        _bedrock_available = False
        app.logger.error("Bedrock client not available (UnknownServiceError). Upgrade boto3/botocore in the image.")
        return None
    except Exception as e:
        # other errors (e.g., misconfigured region)
        _bedrock_available = False
        app.logger.exception("Failed to create bedrock client: %s", e)
        return None

# DynamoDB table for conversation history (optional)
DDB_TABLE = os.environ.get("DDB_TABLE")

# Simple in-memory store for uploaded doc text during a container lifetime (MVP).
# For production, store files in S3 and keep references in DynamoDB.
DOC_STORE = {}

INDEX_HTML = """
<!doctype html>
<title>DocChat</title>
<h1>DocChat — Ask questions about your document</h1>
<form action="/upload" method=post enctype=multipart/form-data>
  <label>Upload plain text file (.txt):</label>
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

<h2>Sample documents in this session</h2>
<ul>
{% for k in docs %}
  <li>{{k}}</li>
{% endfor %}
</ul>
"""

def prompt_for_question(doc_text, question, max_context_chars=3000):
    """
    Build a prompt that includes document context and the question.
    Keep context trimmed to `max_context_chars`.
    """
    ctx = doc_text.strip()
    if len(ctx) > max_context_chars:
        ctx = ctx[-max_context_chars:]  # use recent chunk; or implement smarter chunking
    prompt = (
        "You are a helpful assistant that answers questions about the provided document.\n\n"
        "Document (excerpt):\n"
        f"{ctx}\n\n"
        "Answer the question below concisely. If the answer is not present in the document, say \"I don't know based on the document.\" Do not hallucinate.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return prompt

def call_bedrock(prompt, max_retries=2, backoff_sec=0.5):
    """
    Calls Bedrock runtime using the lazy client factory.
    Returns the raw text response (str).
    Raises RuntimeError if Bedrock client is not available.
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
                body=json.dumps(payload).encode("utf-8")
            )
            body_bytes = resp.get("body").read()
            # --- existing robust parsing logic (copied from your function) ---
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

    # If all retries failed, log and raise
    app.logger.exception("Bedrock invocation failed after %d attempts. Last error: %s", max_retries, last_exc)
    raise last_exc

def save_conversation(doc_id, question, answer):
    if not DDB_TABLE:
        return
    try:
        ddb = boto3.resource("dynamodb")
        table = ddb.Table(DDB_TABLE)
        item = {
            "id": str(uuid.uuid4()),
            "doc_id": doc_id,
            "question": question,
            "answer": answer,
            "ts": int(datetime.utcnow().timestamp())
        }
        table.put_item(Item=item)
    except Exception:
        logger.exception("Failed to save conversation to DynamoDB")

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, docs=list(DOC_STORE.keys()))

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return "No file uploaded", 400
    data = f.read()
    try:
        text = data.decode("utf-8")
    except:
        text = data.decode("latin-1", errors="ignore")
    doc_id = str(uuid.uuid4())[:8]
    DOC_STORE[doc_id] = text
    return f"Uploaded as document id: {doc_id}. Go back and ask a question. <a href='/'>Home</a>"

@app.route("/ask", methods=["POST"])
def ask():
    doc_id = request.form.get("doc_id")
    question = request.form.get("question", "").strip()
    if not doc_id or doc_id not in DOC_STORE:
        return "Invalid or missing doc_id", 400
    if not question:
        return "Question empty", 400

    prompt = prompt_for_question(DOC_STORE[doc_id], question)
    try:
        answer = call_bedrock(prompt)
    except Exception as e:
        return f"Bedrock error: {e}", 500

    # save conversation optionally
    try:
        save_conversation(doc_id, question, answer)
    except:
        pass

    return render_template_string("<h3>Answer</h3><pre>{{answer}}</pre><p><a href='/'>Back</a></p>", answer=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
