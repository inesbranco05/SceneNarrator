import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

def generate_narration(scene_description, environment_context):

    prompt = f"""
    You are a visual assistant for a person.

    Environment memory:
    {environment_context}

    Current scene update:
    {scene_description}

    Describe ONLY what is changing or relevant now.

    Instructions:
    - Use ONLY the provided information.
    - NEVER guess actions.
    - NEVER guess intentions.
    - NEVER guess emotions.
    - NEVER assume activities.
    - Mention only detected objects.
    - Use the environment context only for location awareness.
    - Be concise.
    - Maximum 2 sentences.

    Generate a useful narration.
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

            "temperature": 0.2
        }
    )

    result = response.json()
    if "choices" not in result:
        print(result)
        return "Error generating narration."
    return result["choices"][0]["message"]["content"]