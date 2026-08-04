# Optional container path — Render can deploy this instead of the native
# Python runtime (set "runtime: docker" in render.yaml or pick Docker in the UI).
# Also works on Railway, Fly.io, Cloud Run, or any container host.
FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal; Streamlit + Plotly are pure-Python wheels.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hosts inject the port via $PORT; default to 8501 for plain `docker run`.
ENV PORT=8501
EXPOSE 8501

# Shell form so $PORT expands at runtime.
CMD streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
