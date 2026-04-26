FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir onnx onnxruntime safetensors spandrel

COPY scripts/run_export.py /usr/local/bin/run_export.py
RUN chmod +x /usr/local/bin/run_export.py

WORKDIR /work

ENTRYPOINT ["python", "/usr/local/bin/run_export.py"]
