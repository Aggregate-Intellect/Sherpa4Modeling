import json
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from clevr_qa.llm import LoggedLLM
import os

TOGETHER_AI_BASE_URL = "https://api.together.xyz/v1"


def load_dataset(
    data_folder="data",
    scene_file="CLEVR_val_scenes.json",
    question_file="CLEVR-Humans-val.json",
):
    with open(f"{data_folder}/{scene_file}", "r") as f:
        scenes = json.load(f)["scenes"]

    with open(f"{data_folder}/{question_file}", "r") as f:
        questions = json.load(f)["questions"]

    return scenes, questions


def load_processed_dataset(
    data_folder="data",
    scene_file="CLEVR_val_scenes.json",
    question_file="filtered_questions.jsonl",
):
    with open(f"{data_folder}/{scene_file}", "r") as f:
        scenes = json.load(f)["scenes"]

    with open(f"{data_folder}/{question_file}", "r") as f:
        questions = [json.loads(line) for line in f]

    return scenes, questions


def get_llm(
    llm_type: str, llm_name: str, temperature: float, log_file: str = ""
) -> BaseChatModel:
    if llm_type == "openai":
        llm = ChatOpenAI(model=llm_name, temperature=temperature)
    elif llm_type == "togetherai":
        api_key = os.environ.get("TOGETHERAI_API_KEY")
        llm = ChatOpenAI(
            base_url=TOGETHER_AI_BASE_URL,
            api_key=api_key,
            model=llm_name,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    if log_file:
        llm = LoggedLLM(llm=llm, logger_file=log_file)
    return llm
