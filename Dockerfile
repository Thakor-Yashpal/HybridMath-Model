# ==========================================
# 🧠 HybridMath Solver (FastAPI + PyTorch CUDA)
# ==========================================

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from project root
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    fastapi uvicorn sympy transformers sentencepiece protobuf

# Copy backend and model folders
COPY BackEnd /app/BackEnd
COPY Model /app/Model

# Expose backend port
EXPOSE 8000

# Environment setup
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Start FastAPI with Uvicorn
CMD ["uvicorn", "BackEnd.backend:app", "--host", "0.0.0.0", "--port", "8000"]
