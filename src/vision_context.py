import os
import base64
import requests

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)

print("OPENROUTER loaded:", OPENROUTER_API_KEY is not None)


def encode_image(image_path):

    with open(image_path, "rb") as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def generate_visual_context(image_path):

    image_b64 = encode_image(image_path)

    prompt = """
Analyze this image and create a permanent environment memory.

Ignore:
- people
- animals
- actions
- gestures
- poses
- bottles
- phones
- remotes
- cups
- books
- temporary objects
- movable objects

Only include:
- walls
- doors
- windows
- furniture
- fixed structures
- permanent room features

Return ONLY the following format:

ROOM TYPE:
<one line>

STATIC OBJECTS:
<comma separated list>

LAYOUT:
<one short sentence>

OBSTACLES:
<one short sentence>

Rules:
- Maximum 50 words.
- No explanations.
- No reasoning.
- No comments.
- No text before the sections.
- No text after the sections.
"""
    response = requests.post(

        "https://openrouter.ai/api/v1/chat/completions",

        headers={
            "Authorization":
            f"Bearer {OPENROUTER_API_KEY}",

            "Content-Type":
            "application/json"
        },

        json={

            "model":
            "nvidia/nemotron-nano-12b-v2-vl:free",

            "temperature": 0,

            "max_tokens": 100,

            "messages": [

                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",
                            "text": prompt
                        },

                        {
                            "type": "image_url",

                            "image_url": {

                                "url":
                                f"data:image/jpeg;base64,{image_b64}"

                            }
                        }
                    ]
                }
            ]
        },

        timeout= 20

    )

    result = response.json()

    print("\nVISION RESPONSE:")
    print(result)

    if "choices" not in result:

        print("Using fallback environment memory.")
        print(result)

        return """
ROOM TYPE:
Unknown

STATIC OBJECTS:
Unknown

LAYOUT:
Unknown

OBSTACLES:
Unknown
"""

    content = result["choices"][0]["message"]["content"]

    return content.strip()