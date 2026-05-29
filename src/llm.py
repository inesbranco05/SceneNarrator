import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")


def generate_narration(scene_description):

    prompt = f"""
    You are a visual assistant for a person.

    Describe the scene naturally.

    Rules:
    - Be brief
    - Avoid repeating previous wording
    - Focus on important changes
    - Mention people first
    - Mention interactions if possible
    - Avoid listing objects mechanically

    Scene:
    {scene_description}
    """

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
                    "role": "user",
                    "content": prompt
                }
            ],

            "temperature": 0.7
        }
    )

    result = response.json()
    if "choices" not in result:
        return "Error generating narration."
    return result["choices"][0]["message"]["content"]