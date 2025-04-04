# Base image
FROM ubuntu:20.04

# Install necessary dependencies
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    net-tools \
    iputils-ping \
    nano \
    vim \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependency file first (for caching)
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy the code AFTER installing dependencies
COPY src /app/llm/src

# ✅ Copy scripts directory to /app/llm/scripts
COPY scripts /app/llm/scripts

# Mount the models directory from the host system
VOLUME ["/app/llm/models"]

# Expose Ollama API port
EXPOSE 11434

# Command to run the LLM startup script
CMD ["python3", "/app/llm/scripts/llm_startup.py"]