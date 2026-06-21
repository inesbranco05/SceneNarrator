import requests
import os

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

API_KEY = os.getenv("GROQ_API_KEY")

print("GROQ loaded:", API_KEY is not None)


def generate_narration(scene_description,environment_context):

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",

        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        },

        json={
            "model": "llama-3.1-8b-instant",

            "messages": [
                {
                    "role": "system",
                    "content": """
                       You are a visual accessibility assistant.

Your goal is to provide useful narration for a visually impaired user.

Use:
- the current detected objects
- the environment memory

The environment memory provides context about the place.

Mention the room type at most once.

Do not repeatedly describe walls, ceilings, lamps,
furniture or other static objects.

Focus primarily on the currently detected objects.

STRICT RULES:

- Never invent objects.
- Never invent actions.
- Never invent intentions.
- Never invent emotions.
- Never estimate distance in meters.
- Never assume what a person is doing.
- Never say "near you", "far from you" or similar expressions.
- Never assume the user's position.
- Describe locations relative to the image only.
- Never describe relationships between detected objects.
- Never say an object is above, below, beside or behind another object unless explicitly provided.
- Never use information that is not present.
- Never describe walls, ceilings or furniture unless they help understand object locations.
- Never say a person is standing, sitting, walking or running.
- Never describe objects that are not currently detected.

When an object disappears, say it is no longer visible.

When an object appears, mention that it appeared.

Keep the narration short and natural.

Maximum 20 words.
"""

                },

                {
                    "role": "user",
                    "content": f"""
                        CURRENT SCENE:

                        {scene_description}

                        ENVIRONMENT MEMORY:

                        {environment_context}

                        Create a short and helpful narration for a visually impaired user.

                        Use only the information provided.
                        """
                }
            ],

            "temperature": 0,
            "max_tokens": 30
        }
    )

    result = response.json()

    if "choices" not in result:
        print(result)
        return "Unable to describe scene."

    return result["choices"][0]["message"]["content"]