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

# Copy everything from the local `ui` directory into the container
COPY ./scripts /app/ui/scripts
COPY requirements.txt /app/

# UI  service will be added later, for now just expose a port
EXPOSE 8000

# Run the Gradio app and bind to 0.0.0.0 for external access
CMD ["python3", "/app/ui/scripts/gradio_app.py", "--server_name", "0.0.0.0"]
