from llm import generate_narration

scene = """
I can see:
- a person on the center
- a chair on the right
- a laptop on the center
"""

response = generate_narration(scene)

print(response)