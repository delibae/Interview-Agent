import os
from typing import List

import requests


# Function to get embeddings from Jina API
def get_embeddings(text_chunks: List[str]) -> List[List[float]]:
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('JINA_API_KEY')}",
    }
    payload = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": text_chunks,
    }
    response = requests.post(url, headers=headers, json=payload)
    response_data = response.json()
    return [item["embedding"] for item in response_data["data"]]
