# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy files
COPY . /app/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        supervisor \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy supervisord config
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Expose FastAPI + Streamlit
EXPOSE 8000 8501

CMD ["/usr/bin/supervisord"]