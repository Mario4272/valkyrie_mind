import gradio as gr
import requests
import json

OLLAMA_API_URL = "http://llm:11434/api/generate"  # Correct Ollama endpoint

def get_model_response(input_text):
    payload = {
        "model": "dolphin-mistral",  # Change this to your actual Ollama model name
        "prompt": input_text,
        "format": "json",  # Ensure the response is in JSON format
        "stream": False,   # Do not stream the response
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()  # Raise error for non-200 responses

        # Debugging the raw response
        print("Raw response from API:", response.text)
        
        # Try parsing the JSON response
        try:
            response_json = response.json()
            # Extracting just the response part
            model_response = response_json.get("response", "Error: No response from model")
        except json.JSONDecodeError as e:
            # Log the error if we can't parse the response
            print("Error decoding JSON:", e)
            model_response = f"Error decoding JSON: {e}"

        return model_response

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Launch Gradio on all interfaces so it's accessible externally
iface = gr.Interface(
    fn=get_model_response,
    inputs="text",
    outputs="text"
).launch(server_name="0.0.0.0", server_port=7860)
