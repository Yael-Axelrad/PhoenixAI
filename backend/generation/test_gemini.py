
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Locate the project root independently of the terminal location.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the API key from the local .env file.
load_dotenv(PROJECT_ROOT / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")

# Create the Gemini client using the secret API key.
client = genai.Client(api_key=api_key)

# Send a simple request to verify that the connection works.
response = client.models.generate_content(
model="gemini-3.8-flash",
    contents="ענה בעברית במשפט אחד: מהו ביטוח רכב?",
)

print("\nGemini response:")
print(response.text)
