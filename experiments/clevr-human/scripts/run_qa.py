from argparse import ArgumentParser
from clevr_qa.question_answerer import get_answerer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool
from loguru import logger

logger.remove()  # remove the default handler configuration
logger.add("logs/run.log", level="INFO", serialize=False)


def process_sample(args):
    sample, answerer, idx = args
    question = sample["question"]
    scene = sample["scene"]
    answer = answerer.answer_question(question, scene, idx)
    return answer


def main(args):
    dataset = load_dataset(args.dataset_name)["train"]
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

    gt_answer = [sample["answer"] for sample in dataset]
    df = pd.DataFrame({"predicted": results, "actual": gt_answer})
    df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name", type=str, default="Dogdays/clevr_subset", help="Dataset name"
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["direct", "state_machine"],
        default="direct",
        help="Approach to answer the question",
    )
    parser.add_argument(
        "--output_file",
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
        choices=["openai", "togetherai"],
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
        default=10,
        help="Number of hops when running the state machine",
    )
    parser.add_argument(
        "--use_scene",
        action="store_true",
        help="Whether to use the scene information in the prompt",
    )

    args = parser.parse_args()
    main(args)
