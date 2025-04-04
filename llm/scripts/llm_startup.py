import os
import time
import subprocess

LLM_PORT = "11434"
MODEL_NAME = "dolphin-mistral:latest"
MODEL_PATH = "/root/.ollama/models"  # Path where models are stored

def install_ollama():
    """Installs Ollama if it's not already installed."""
    try:
        print("Checking for Ollama installation...")
        # Check if ollama is installed
        subprocess.run(["which", "ollama"], check=True)
        print("Ollama is already installed.")
    except subprocess.CalledProcessError:
        print("Ollama not found, installing...")
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

def check_if_model_exists(model_name):
    """Checks if the model already exists in the local directory."""
    model_path = os.path.join(MODEL_PATH, model_name)
    if os.path.exists(model_path):
        print(f"Model {model_name} found locally.")
        return True
    else:
        print(f"Model {model_name} not found locally.")
        return False

def start_llm_server():
    """Starts the LLM server using ollama."""
    print("Starting llm server...")
    subprocess.Popen(["ollama", "serve"])

    # Wait for the server to start
    time.sleep(5)

def load_model():
    """Loads the default model into llm, or pulls if necessary."""
    print(f"Loading model: {MODEL_NAME}")
    
    # Check if the model is present locally
    if not check_if_model_exists(MODEL_NAME):
        print("Model not found locally. Pulling model...")
        try:
            subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to pull model: {MODEL_NAME}")
    
    # Run the model
    try:
        subprocess.run(["ollama", "run", MODEL_NAME], check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to load model: {MODEL_NAME}")

if __name__ == "__main__":
    # Ensure Ollama is installed
    install_ollama()
    
    # Start the LLM server and load the model
    start_llm_server()
    load_model()

    print(f"llm is running on port {LLM_PORT}")
    
    # Keep container alive
    while True:
        time.sleep(60)
