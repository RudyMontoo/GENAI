from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv
import time

load_dotenv()

# ✅ Model
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# ✅ Structured Output
class FeedbackResponse(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment")
    reply: str = Field(description="Appropriate response to the feedback")

parser = PydanticOutputParser(pydantic_object=FeedbackResponse)

# ✅ Single Prompt (classification + response together)
prompt = PromptTemplate(
    template="""
Analyze the feedback and do two things:
1. Classify sentiment as positive or negative
2. Generate an appropriate response

Feedback:
{feedback}

{format_instruction}
""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    }
)

# ✅ Chain
chain = prompt | model | parser

# ✅ Retry wrapper (handles 429 / 503)
def safe_invoke(input_data, retries=3):
    for i in range(retries):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            print(f"Retry {i+1} due to error: {e}")
            time.sleep(5 * (i + 1))
    return "Failed after retries"

# ✅ Run
result = safe_invoke({
    "feedback": "This is a beautiful phone"
})

print(result)