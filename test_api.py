import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load the .env file
load_dotenv()

# 2. Initialize the client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    # 3. Simple test call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Connection Successful'"}],
        max_tokens=10
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")