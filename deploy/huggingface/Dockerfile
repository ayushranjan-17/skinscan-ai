FROM python:3.11-slim

# libglib is needed by OpenCV at import time
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user \
    && mkdir -p /home/user/app/data \
    && chown -R user:user /home/user
USER user
ENV PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=user backend backend
COPY --chown=user ml_pipeline ml_pipeline
COPY --chown=user models models

ENV DEVICE=cpu \
    SKINSCAN_DB_PATH=/home/user/app/data/skinscan.db \
    PYTHONUNBUFFERED=1

EXPOSE 7860

# Without SKINSCAN_AUTH_SECRET set as a Space secret, use a random per-start secret
# rather than the default in the source, which is public.
CMD ["sh", "-c", "export SKINSCAN_AUTH_SECRET=\"${SKINSCAN_AUTH_SECRET:-$(python -c 'import secrets; print(secrets.token_hex(32))')}\" && exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
