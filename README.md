# 2026-ei-aoopii-b12

## Group Members
- Inês Branco - nº 28990
- Sara Rodrigues - nº 28902

## Topic Proposal
Scene Narrator

## Track

Track A – Deep Learning

## Project Description

Imagine pointing a camera at your surroundings and instantly hearing a clear description of what is happening around you.

Scene Narrator is a real-time visual accessibility assistant designed to improve environmental awareness for visually impaired users. By combining computer vision, deep learning and large language models, the system is able to understand a scene and transform visual information into natural spoken descriptions.

## Tech Stack
Python 3, YOLOv8 (Ultralytics), Groq API (Llama 3.1 8B Instant), OpenRouter API, NVIDIA Nemotron Nano 12B Vision Language Model, OpenCV, pyttsx3, requests, python-dotenv

## How to Run

1. Clone the repository
```bash
git clone <repository-url>
cd 2026-EI-AOOPII-B12
```

2. Create and activate a virtual environment

```bash
git clone <repository-url>
cd 2026-EI-AOOPII-B12
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Create a .env file in the project root
```bash
GROQ_API_KEY=your_groq_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

5. Run the application
```bash
cd src
python detector.py
```

6. Controls
- Press Q to exit the application.
- The webcam must be available and connected.
- An internet connection is required for LLM and vision model requests.

## System Architecture

Webcam
→ YOLOv8 Object Detection
→ Scene Analysis
→ Environment Context Generation
→ LLM Narration
→ Text-to-Speech Output

The environment memory is generated once at startup and is used as contextual information during scene narration.

## Limitations

The environment memory relies on a free vision-language model accessed through OpenRouter.

Due to rate limits or temporary service unavailability, environment context generation may occasionally fail. In such cases, the application falls back to real-time object detection and narration without contextual environment information.

## Conclusion

Scene Narrator demonstrates how Deep Learning, Computer Vision and Large Language Models can be combined to create a real-time accessibility assistant.
By integrating YOLOv8 object detection, environment context generation and speech synthesis, the system is capable of providing auditory descriptions of the user's surroundings. The project highlights the potential of AI-based solutions to improve environmental awareness for visually impaired users.
Future improvements could include automatic environment context updates when changing locations, more accurate spatial reasoning and support for additional object detection models.