FROM python:3.11-slim

# Install system dependencies for OpenCV, multimedia, and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files
COPY backend/requirements.txt /app/backend/requirements.txt
COPY cv-pipeline/requirements.txt /app/cv-pipeline/requirements.txt
COPY nlp-pipeline/requirements.txt /app/nlp-pipeline/requirements.txt

# Install dependencies (CPU-optimized PyTorch by default for container portability)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application codebase
COPY . /app

# Expose backend port
EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
