# FROM public.ecr.aws/docker/library/python:3.13-slim
# WORKDIR /app

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 80
# CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app", "--workers", "2"]


FROM public.ecr.aws/docker/library/python:3.13-slim
WORKDIR /app

# install system deps needed by some packages (optional, just in case)
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# enforce upgrade to ensure fresh boto3/botocore
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app", "--workers", "2"]

