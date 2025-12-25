FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install mlflow

COPY src/ src/
COPY scripts/ scripts/
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

EXPOSE 8000
EXPOSE 5000

ENTRYPOINT ["./entrypoint.sh"]

