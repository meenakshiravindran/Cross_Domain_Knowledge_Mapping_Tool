# ---------------------------
# Minimal Dockerfile
# ---------------------------

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by spaCy + sentence-transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (cache optimization)
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Register spaCy model (needed even after installing from .whl)
RUN python -m spacy validate

# Copy your full project
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
