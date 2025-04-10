# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_model(model_name:str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

