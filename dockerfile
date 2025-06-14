FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# 1) System deps for FAISS and cleanup
RUN apt-get update \
 && apt-get install -y libgomp1 build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy code + data files
COPY requirements.txt main_api.py embeddings.npz /app/

# 3) Install Python deps
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir faiss-cpu

# 4) Expose & launch
EXPOSE 8080
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8080"]
