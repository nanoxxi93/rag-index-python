import os
from dotenv import load_dotenv
from llama_index.llms.nvidia import NVIDIA

load_dotenv()
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY', "")

# Probar diferentes modelos
models_to_test = [
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mistral-large-2-123b-instruct",
]

for model in models_to_test:
    try:
        print(f"Testing {model}...")
        llm = NVIDIA(
            model=model,
            api_key=NVIDIA_API_KEY,
            timeout=5.0  # Timeout corto para prueba
        )
        response = llm.complete("Hello, test")
        print(f"✅ {model} is available")
        print(f"Response: {response[:50]}...")
        break
    except Exception as e:
        print(f"❌ {model} failed: {str(e)[:100]}")