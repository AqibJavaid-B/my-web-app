FROM public.ecr.aws/docker/library/python:3.13-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 80
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app", "--workers", "2"]
