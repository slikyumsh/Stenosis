FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PIP_DEFAULT_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.app.txt /app/requirements.app.txt
RUN pip install --no-cache-dir -r /app/requirements.app.txt

COPY app /app/app
COPY best.onnx /app/best.onnx

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
