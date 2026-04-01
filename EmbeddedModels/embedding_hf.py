from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vector=embedding.embed_query("Delhi is the capital of India")
print(str(vector))