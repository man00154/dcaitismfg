# Use official Python base image
FROM python:3.10-slim

# System deps (faiss wheel works; libgomp1 improves BLAS perf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Avoid pyc files and force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app
COPY app.py /app/app.py

# (Optional) env var for the API key; you can also use Streamlit secrets locally via TOML
# ENV GEMINI_API_KEY=your_key_here

# Expose default Streamlit port
EXPOSE 8501

# Streamlit config to be friendlier in containers
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run
CMD ["streamlit", "run", "app.py"]
