from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq

load_dotenv()

print("Key:", os.getenv("GROQ_API_KEY"))

llm = ChatGroq(model="llama-3.3-70b-versatile")
print(llm.invoke("Say hello").content)