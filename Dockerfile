# ================================
# DOCKERFILE — MODEL API
# ================================

# 1. Base image
FROM python:3.12-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy API code
COPY serve_fastapi.py .

# 7. Copy trained model folder
COPY best_model ./best_model

# 8. Expose FastAPI port
EXPOSE 8000

# 9. Run API on container start
CMD ["uvicorn", "serve_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
