FROM python:3.10-slim

WORKDIR /app

# System deps (important for numpy/torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install numpy FIRST (critical)
RUN pip install --no-cache-dir numpy==1.26.4

# Install CPU PyTorch (stable for ARM)
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]