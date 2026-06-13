import os
import base64
import requests

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)


def encode_image(image_path):

    with open(image_path, "rb") as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def generate_visual_context(image_path):

    image_b64 = encode_image(image_path)

    prompt = """
    Analyze this image and create a permanent environment memory.

    IMPORTANT:

    - Ignore people.
    - Ignore animals.
    - Ignore temporary actions.
    - Ignore gestures.
    - Ignore poses.

    Describe ONLY:

    ROOM TYPE:
    (one line)

    STATIC OBJECTS:
    (list)

    LAYOUT:
    (short description)

    OBSTACLES:
    (short description)

    Maximum 100 words.

    Return ONLY these sections.
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
            "meta-llama/llama-3.2-11b-vision-instruct",

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
        }
    )

    result = response.json()

    print("\nVISION RESPONSE:")
    print(result)

    return result[
        "choices"
    ][0]["message"]["content"]