# Tactical Edge Vision System - Multi-arch CPU Image
# linux/amd64 + linux/arm64 (reliable first; GPU later)
FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 curl \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Bake yolo11n.pt for offline / air-gapped use
RUN python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

COPY vision_node.py .
ENV VIDEO_SOURCE=0 CONF_THRESHOLD=0.45 MODEL_PATH=yolo11n.pt PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "vision_node.py"]