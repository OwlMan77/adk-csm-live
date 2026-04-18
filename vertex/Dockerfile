FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    torchaudio \
    omnivoice \
    soundfile \
    fastapi \
    "uvicorn[standard]" \
    "huggingface_hub[hf_transfer]"

ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Download model weights at build time
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

COPY vertex/download_model.py .
RUN python download_model.py

# Bake in reference voice samples (audio1-5 .wav + .txt pairs)
COPY scripts/samples /app/samples

COPY vertex/server.py .

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
