import json
import os
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether

from clevr_qa.llm import LoggedLLM


class JsonOutputParser(BaseOutputParser):
    def parse(self, text: str) -> dict:
        json_pattern = re.compile(r"```json\n((.|\n)*?)\n```")
        try:
            matches = json_pattern.findall(text)
            if matches:
                result = json.loads(matches[-1][0])
                return result
            else:
                return json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse JSON from text: {text}")

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
    elif llm_type == "together":
        llm = ChatTogether(model=llm_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")

    llm = LoggedLLM(llm=llm, logger_file=log_file)
    return llm
