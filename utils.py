from google import genai
from groq import Groq
import os
from dotenv import load_dotenv
import requests

def to_float_or_default(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return 1.2

def format_text(song: str, artist: str):
    prompt = f"Provide a sentiment score for the song “{song}” by {artist} based on its lyrics and musical features. Use a scale from 0 to 1, where 0 represents very sad and 1 represents very happy. Return only the numeric value with no additional text."

    try:
        response_val = use_groq(prompt)
        return to_float_or_default(response_val)
    except Exception:
        pass

    # Try Gemini
    try:
        response_val = use_gemini(prompt)
        return to_float_or_default(response_val)
    except Exception:
        pass

    return 1.2


def use_gemini(prompt: str):
    load_dotenv()
    key= os.getenv("mnd_gemini_key")
    client = genai.Client(api_key=key)
    # Generate text
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text


def use_groq(prompt: str):
    load_dotenv()
    key= os.getenv("mnd_groq_key")
    client = Groq(api_key=key)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role":"user", "content": prompt}
        ]
    )
    return response.choices[0].message.content