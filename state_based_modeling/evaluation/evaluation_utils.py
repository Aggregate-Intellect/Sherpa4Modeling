import json
from multiprocessing import Pool
from typing import Dict, List

import numpy as np
from tqdm import tqdm
from worde4mde import load_embeddings

from evaluation.model_matcher import Grader
from evaluation.model_metadata import DomainModel

sgram_mde = load_embeddings("sgram-mde")


def evaluate_one_domain(args) -> dict:
    model_path, reference_path, embedding_path = args

    grader = Grader(sgram_mde)
    reference_model = DomainModel(reference_path)
    model = DomainModel(model_path, transform=True)

    result = grader.calculate_scores(reference_model, model, embedding_path)
    return result


def get_evaluation_results(
    model_folder: str, reference_folder: str, domains: List[str], num_processes: int = 8
) -> Dict[str, float]:
    data = []

    for domain in domains:
        model_path = f"{model_folder}/{domain}.txt"
        reference_path = f"{reference_folder}/{domain}.txt"
        embedding_caching_path = f"{model_folder}/{domain}_ada_embedding.txt"
        data.append((model_path, reference_path, embedding_caching_path))

    with Pool(num_processes) as p:
        results = list(p.imap(evaluate_one_domain, data))

    for i, domain in enumerate(domains):
        if "direct" in model_path:
            results[i]["num_llm_calls"] = 1
            continue

        domain_data_file = f"{model_folder}/{domain}.json"
        with open(domain_data_file, "r") as file:
            domain_data = json.load(file)
            results[i]["num_llm_calls"] = domain_data["llm_calls"]

    return {
        "precision": np.mean([r["class"]["precision"] for r in results]),
        "recall": np.mean([r["class"]["recall"] for r in results]),
        "f1": np.mean([r["class"]["f1"] for r in results]),
        "num_llm_calls": np.mean([r["num_llm_calls"] for r in results]),
    }
