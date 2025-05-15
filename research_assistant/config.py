import os 
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
DEFAULT_MODEL= "llama-3.3-70b-versatile"


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_RETRIEVAL_K = 3

def init_environment():
    """Initialize environment varaible"""
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    os.environ["HF_TOKEN"] = HF_TOKEN