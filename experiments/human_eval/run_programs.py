from argparse import ArgumentParser
from direct_prompt import generate_programs as generate_programs_direct
from state_machine.execution import generate_programs as generate_programs_state_machine
from loguru import logger


def main(args):
    logger.info(f"Running with args: {args}")
    if args.method == "state_machine":
        generate_programs_state_machine(args)
    elif args.method == "direct_prompt":
        generate_programs_direct(args)

    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--llm_family", type=str,
                        choices=["openai", "togetherai"], default="openai")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--output_filename", type=str,
                        default="state_machine_samples.jsonl")
    parser.add_argument("--saving_frequency", type=int, default=5)
    parser.add_argument("--method", type=str,
                        choices=["state_machine", "direct_prompt"], default="state_machine")

    args = parser.parse_args()
    main(args)
