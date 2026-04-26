FROM docker.io/pinto0309/onnx2tf:2.3.14

USER root

ENV PIP_BREAK_SYSTEM_PACKAGES=1

# tensorflowjs 4.x depends on tensorflow_decision_forests at import time; that package pins TF versions
# that conflict with the TF in this image on some platforms. Stub TF-DF for SavedModel-only conversion.
RUN pip install --no-cache-dir -U --no-deps tensorflowjs==4.22.0 tensorflow_hub==0.16.1 \
  && pip install --no-cache-dir "jax>=0.4.13" "jaxlib>=0.4.13"

COPY docker/convert/stubs/tensorflow_decision_forests \
  /usr/local/lib/python3.12/dist-packages/tensorflow_decision_forests

COPY scripts/run_convert.sh /usr/local/bin/run_convert.sh

WORKDIR /work

ENTRYPOINT ["bash", "/usr/local/bin/run_convert.sh"]
