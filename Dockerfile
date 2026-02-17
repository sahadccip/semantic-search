FROM ubuntu:22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Pull required models
RUN ollama pull nomic-embed-text
RUN ollama pull llama3

# Expose port
EXPOSE 10000

# Start everything
CMD ollama serve & uvicorn app:app --host 0.0.0.0 --port 10000
