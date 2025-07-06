import os
import sys
from argparse import ArgumentParser

import dotenv
from llm_coder.utils import generate_programs
from loguru import logger

dotenv.load_dotenv()

logger.remove()  # remove the default handler configuration
logger.add(sys.stderr, level=os.environ.get(
    "LOG_LEVEL", "INFO"), serialize=False)


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
    parser.add_argument("--num_parallel", type=int, default=1)
    parser.add_argument("--method", type=str,
                        choices=["state_machine", "state_machine_with_feedback", "direct_prompt", "agent_coder", "agent_coder_improved"], default="state_machine")

    args = parser.parse_args()
    main(args)
