import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

# Load env
load_dotenv()

# Initialize LLM (NOT chat model)
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",   # or gemini-1.5-pro
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# Simple prompt (no messages, just text)
prompt = "Analyze a startup based on its GitHub activity and provide a risk score."

response = llm.invoke(prompt)

print(response)



# THIS CODE WILL NOT WORK BECAUSE GOOGLE LLM MODEL IS NOT WORKING PROPERLY CURRENTLY BUT CHAT MODEL IS OWRKING 
