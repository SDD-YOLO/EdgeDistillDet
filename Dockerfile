# EdgeDistillDet API / Worker（CPU 基础镜像；GPU 训练需在主机安装 NVIDIA Container Toolkit 并使用带 CUDA 的基础镜像替换）
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml README.md LICENSE ./
COPY main.py ./
COPY core ./core
COPY utils ./utils
COPY scripts ./scripts
COPY configs ./configs
COPY web ./web
COPY tests ./tests

RUN pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "-u", "web/app.py"]
