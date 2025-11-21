import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

def chat():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    model = "gemini-2.0-flash"

    print("Chatbot ready! Type 'exit' to quit.\n")

    while True:
        user_message = input("You: ")

        if user_message.lower() == "exit":
            break

        contents = [
            types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            )
        ]

        generate_content_config = types.GenerateContentConfig()

        print("Bot: ", end="")
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config
        ):
            if chunk.text:
                print(chunk.text, end="")
        print("\n")

if __name__ == "__main__":
    chat()
