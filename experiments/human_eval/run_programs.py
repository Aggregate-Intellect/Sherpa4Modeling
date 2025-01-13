from argparse import ArgumentParser
from loguru import logger
from llm_coder.utils import generate_programs


def main(args):
    logger.info(f"Running with args: {args}")
    generate_programs(args)

    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--llm_family", type=str,
                        choices=["openai", "togetherai"], default="openai")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output_filename", type=str,
                        default="state_machine_samples.jsonl")
    parser.add_argument("--saving_frequency", type=int, default=30)
    parser.add_argument("--num_parallel", type=int, default=8)
    parser.add_argument("--method", type=str,
                        choices=["state_machine", "direct_prompt"], default="state_machine")

    args = parser.parse_args()
    main(args)
