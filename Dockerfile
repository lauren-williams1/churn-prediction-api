# Serving image: FastAPI + exported sklearn pipeline (no MLflow server needed at runtime).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /srv

# Create non-root user first
RUN useradd --create-home appuser

# Dependencies first so Docker caches this layer when only code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Copy application files
COPY app/ app/
COPY models/pipeline.joblib models/metadata.json models/

# Grant appuser ownership of the entire /srv directory
RUN chown -R appuser:appuser /srv

# Switch to non-root user
USER appuser

EXPOSE 8000

# Container-level health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]