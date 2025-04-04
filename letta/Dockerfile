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

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /app

# Copy the code
COPY . /app

# Letta service will be added later, for now just expose a port
EXPOSE 8000

# Placeholder command for Letta service
CMD ["echo", "Letta service placeholder"]
