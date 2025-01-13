from __future__ import annotations

from human_eval.data import read_problems, write_jsonl
import os
import json
from loguru import logger
import numpy as np
from typing import TYPE_CHECKING
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm
from langchain_openai import ChatOpenAI

from llm_coder.coders.direct_prompt_coder import DirectPromptCoder
from llm_coder.coders.test_based_sm_coder import TestBasedSMCoder

if TYPE_CHECKING:
    from llm_coder.coders.base import BaseCoder


TOGETHER_AI_BASE_URL = "https://api.together.xyz/v1"


def get_coder(method: str, llm_family: str, model_name: str, temperature: float = 0.01):
    llm = get_llm(llm_family, model_name, temperature)

    if method == "direct_prompt":
        return DirectPromptCoder(llm=llm)
    elif method == "state_machine":
        return TestBasedSMCoder(llm=llm)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_llm(llm_family, model_name, temperature):
    if llm_family == "openai":
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif llm_family == "togetherai":
        api_key = os.environ.get("TOGETHERAI_API_KEY")
        return ChatOpenAI(
            base_url=TOGETHER_AI_BASE_URL,
            api_key=api_key,
            model=model_name,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown LLM family: {llm_family}")


def load_cache(filename: str) -> list[dict]:
    cache_results = []
    if not os.path.exists(filename):
        return cache_results

    with open(filename, "r") as f:
        for line in f:
            cache_results.append(json.loads(line))

    return cache_results


def generate_one_solution(config) -> dict:
    args, task_id, problem = config
    coder = get_coder(args.method, args.llm_family,
                      args.llm_model, args.temperature)
    result = {
        "task_id": task_id,
        "completion": coder.solve_problem(problem)
    }

    return result


def generate_programs(args):
    problems = read_problems()
    task_ids = list(problems.keys())

    results = load_cache(args.output_filename)
    processed_task_ids = [s["task_id"] for s in results]
    task_ids = [id for id in task_ids if id not in processed_task_ids]

    logger.info(
        f"Generate programs for {len(task_ids)} tasks with {
            args.saving_frequency} batch size and {args.num_parallel} parallel processes"
    )

    num_batches = int(np.ceil(len(task_ids) / args.saving_frequency))
    for batch_id in tqdm(range(num_batches), desc="Processing batches"):
        batch_task_ids = task_ids[batch_id *
                                  args.saving_frequency: (batch_id + 1) * args.saving_frequency]

        with Pool(args.num_parallel) as p:
            results_batch = p.map(
                generate_one_solution,
                [(args, task_id, problems[task_id])
                 for task_id in batch_task_ids]
            )

        results.extend(results_batch)
        write_jsonl(args.output_filename, results)
