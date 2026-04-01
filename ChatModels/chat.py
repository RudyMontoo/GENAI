import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the LLM with a current stable model
# gemini-2.5-flash is the successor to 1.5-flash
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

try:
    # Invoke the model with a simple prompt
    result = model.invoke("What is the capital of India?")
    print(result.content)
except Exception as e:
    # Print the specific error if the call fails
    print(f"Error: {e}")