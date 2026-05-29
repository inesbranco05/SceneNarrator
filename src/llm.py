import requests
import os

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

def generate_environment_context(scene_description):

    prompt = f"""
    You are analyzing the initial environment
    for a blind assistance system.

    Create a concise environmental memory.

    Focus on:
    - type of place
    - important static objects
    - general layout

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

            "temperature": 0.4
        }
    )

    result = response.json()

    if "choices" not in result:
        return "Unknown environment."

    return result["choices"][0]["message"]["content"]

def generate_narration(scene_description, environment_context):

    prompt = f"""
    You are assisting a person.

    Environment memory:
    {environment_context}

    Current scene update:
    {scene_description}

    Describe ONLY what is changing or relevant now.

    Rules:
    - Be brief
    - Avoid repeating environment details
    - Focus on movement and changes
    - Mention people first
    - Sound natural
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