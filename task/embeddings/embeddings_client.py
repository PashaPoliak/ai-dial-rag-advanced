import json

import requests

DIAL_EMBEDDINGS = 'https://ai-proxy.lab.epam.com/openai/deployments/{model}/embeddings'


#TODO:
# ---
# https://dialx.ai/dial_api#operation/sendEmbeddingsRequest
# ---
# Implement DialEmbeddingsClient:
# - constructor should apply deployment name and api key
# - create method `get_embeddings` that will generate embeddings for input list (don't forget about dimensions)
#   with Embedding model and return back a dict with indexed embeddings (key is index from input list and value vector list)

class DialEmbeddingsClient:
    def __init__(self, deployment_name: str, api_key: str):
        self.deployment_name = deployment_name
        self.api_key = api_key
        self.endpoint = DIAL_EMBEDDINGS.format(model=deployment_name)

    def get_embeddings(self, inputs: list[str], dimensions: int = 1536) -> dict[int, list[float]]:
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json"
        }

        request_data = {
            "input": inputs,
            "dimensions": dimensions
        }

        response = requests.post(url=self.endpoint, headers=headers, json=request_data, timeout=60)

        if response.status_code == 200:
            data = response.json()
            embeddings_result = {}
            for item in data.get("data", []):
                index = item["index"]
                embedding = item["embedding"]
                embeddings_result[index] = embedding
            return embeddings_result
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")


# Hint:
#  Response JSON:
#  {
#     "data": [
#         {
#             "embedding": [
#                 0.19686688482761383,
#                 ...
#             ],
#             "index": 0,
#             "object": "embedding"
#         }
#     ],
#     ...
#  }
