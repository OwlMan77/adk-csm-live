"""OmniVoice speech synthesis via Vertex AI Model Garden.

The endpoint accepts plain-text instances and returns base64-encoded WAV audio.
The reference voice is fixed server-side (baked into the container at build time).

Request:  {"instances": [{"text": "..."}]}
Response: {"predictions": [{"audio": "<base64-wav>"}]}
"""

import asyncio
import base64
import os
from functools import lru_cache

import google.cloud.aiplatform as aiplatform


@lru_cache(maxsize=1)
def _get_endpoint() -> aiplatform.Endpoint:
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.environ.get("OMNIVOICE_LOCATION", "us-central1")
    endpoint_id = os.environ["OMNIVOICE_ENDPOINT_ID"]

    aiplatform.init(project=project, location=location)
    return aiplatform.Endpoint(
        endpoint_name=f"projects/{project}/locations/{location}/endpoints/{endpoint_id}"
    )


async def synthesize(text: str) -> bytes:
    """Synthesize text via OmniVoice and return WAV bytes."""
    endpoint = _get_endpoint()
    instances = [{"text": text, "steps": 16 }]

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: endpoint.predict(instances=instances),
    )

    audio_b64: str = response.predictions[-1]["audio"]
    return base64.b64decode(audio_b64)
