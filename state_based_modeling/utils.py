from langchain_together import ChatTogether
from sherpa_ai.models import SherpaChatOpenAI


def get_llm(type: str, model_name: str):
    if type == "openai":
        return SherpaChatOpenAI(model_name=model_name, temperature=0.01)
    elif type == "together":
        return ChatTogether(model=model_name, temperature=0.01)
    else:
        raise ValueError(f"Unknown model type {type}")
