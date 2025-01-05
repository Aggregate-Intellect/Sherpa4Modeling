import json
from human_eval.data import read_problems, write_jsonl
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from tqdm import tqdm
from parsers import PythonOutputParser
import os
from loguru import logger

TOGETHER_AI_BASE_URL = "https://api.together.xyz/v1"

PROMPT_TEMPLATE = """Complete the following code. Use ```python to put the completed Python code, including the necessary imports, in markdown quotes:\n{problem}"""


class DirectPromptCoder(BaseModel):
    llm: BaseChatModel
    parser: PythonOutputParser = PythonOutputParser()

    def solve_problem(self, problem):
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | self.llm | self.parser
        result = None
        while result is None:
            result = chain.invoke({
                "problem": problem
            })
        return result


def get_openai_coder(model_name: str, temperature: float = 0.01):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    coder = DirectPromptCoder(llm=llm)

    return coder


def load_cache(filename: str) -> list[dict]:
    cache_results = []
    if not os.path.exists(filename):
        return cache_results

    with open(filename, "r") as f:
        for line in f:
            cache_results.append(json.loads(line))

    return cache_results


def get_togetherai_coder(model_name: str, temperature: float = 0.01):
    api_key = os.environ.get("TOGETHERAI_API_KEY")
    llm = ChatOpenAI(
        base_url=TOGETHER_AI_BASE_URL,
        api_key=api_key,
        model=model_name,
        temperature=temperature
    )
    coder = DirectPromptCoder(llm=llm)

    return coder


def get_coder(llm_family: str, model_name: str, temperature: float = 0.01):
    if llm_family == "openai":
        return get_openai_coder(model_name, temperature)
    elif llm_family == "togetherai":
        return get_togetherai_coder(model_name, temperature)
    else:
        raise ValueError(f"Unknown LLM family: {llm_family}")


def generate_programs(args):
    problems = read_problems()

    task_ids = list(problems.keys())

    coder = get_coder(args.llm_family, args.llm_model, args.temperature)

    results = load_cache(args.output_filename)
    processed_task_ids = [s["task_id"] for s in results]
    task_ids = [id for id in task_ids if id not in processed_task_ids]

    for i, task_id in enumerate(tqdm(task_ids)):
        result = {
            "task_id": task_id,
            "completion": coder.solve_problem(problems[task_id]["prompt"])
        }
        results.append(result)

        if i % args.saving_frequency == 0:
            write_jsonl(args.output_filename, results)
    write_jsonl(args.output_filename, results)
