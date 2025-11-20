# pip install google-genai

import os
from google import genai
from google.genai import types

def generate():
    # Load the API key from environment variable
    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY")
    )

    model = "gemini-2.0-pro"   # Or gemini-2.0-pro if available

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text("INSERT_INPUT_HERE"),
            ],
        )
    ]

    # Example: enabling Google Search as a tool
    tools = [
        types.Tool(
            googleSearch=types.GoogleSearch()
        )
    ]

    # Correct GenerateContentConfig syntax
    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        thinking_config=types.ThinkingConfig(
            thinking_level=types.ThinkingConfig.ThinkingLevel.HIGH,
        ),
    )

    # Streaming response
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config
    ):
        if chunk.text:
            print(chunk.text, end="")

if __name__ == "__main__":
    generate()
