import os
from argparse import ArgumentParser
from multiprocessing import Pool

import dotenv
import pandas as pd
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm

from clevr_qa.question_answerer import get_answerer

dotenv.load_dotenv()  # Load environment variables from .env file

logger.remove()  # remove the default handler configuration
logger.add("logs/run.log", level="INFO", serialize=False, mode="w")


def process_sample(args):
    sample, answerer, idx = args
    question = sample["question"]
    scene = sample["scene"]
    try:
        answer, count = answerer.answer_question(question, scene, idx)
    except Exception as e:
        logger.error(f"Error processing sample {idx}: {e}")
        raise e
    return answer, count


def main(args):
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    dataset = load_dataset(args.dataset_name)["train"]
    # dataset = dataset.select(range(4, 5))
    answerer = get_answerer(args)
    results = []
    with Pool(args.num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(
                    process_sample,
                    [(sample, answerer, idx) for idx, sample in enumerate(dataset)],
                ),
                total=len(dataset),
            )
        )

    answers = [res[0] for res in results]
    counts = [res[1] for res in results]
    output_file = os.path.join(args.output_folder, f"{args.approach}.csv")
    gt_answer = [sample["answer"] for sample in dataset]
    df = pd.DataFrame(
        {"predicted": answers, "actual": gt_answer, "num_llm_calls": counts}
    )
    df.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="Dogdays/clevr_subset", help="Dataset name"
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["direct", "state_machine", "routing", "react"],
        default="direct",
        help="Approach to answer the question",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output file to save the results",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=16,
        help="Number of processes to use for parallel processing of the dataset",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="logs",
        help="Folder to save the logs of the language model",
    )

    # Arguments for LLMs
    parser.add_argument(
        "--llm_type",
        type=str,
        choices=["openai", "together"],
        default="openai",
        help="Provider of the language model",
    )
    parser.add_argument(
        "--llm_name",
        type=str,
        default="gpt-4o-mini",
        help="Name of the language model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Temperature of the language model",
    )

    # Arguments for the state machine methods
    parser.add_argument(
        "--num_runs",
        type=int,
        default=20,
        help="Number of hops when running the state machine",
    )
    parser.add_argument(
        "--use_scene",
        action="store_true",
        help="Whether to use the scene information in the prompt",
    )

    args = parser.parse_args()
    main(args)
