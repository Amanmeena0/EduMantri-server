from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
import os


load_dotenv()
HF_TOKEN= os.getenv("HF_TOKEN")
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY") 
os.environ["GOOGLE_API_KEY"] = GOOGLE_KEY


class Model:
    def get_embeddings():
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en",
            model_kwargs={"device": "cpu"}, 
            encode_kwargs={"batch_size": 64, "normalize_embeddings": True}
        )
        

    def get_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.2,
            top_p=0.9,
            google_api_key=GOOGLE_KEY
        )
       
get_embedding = Model.get_embeddings()
get_llms = Model.get_llm()