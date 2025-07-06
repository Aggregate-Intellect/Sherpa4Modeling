import argparse
import json
import os
from multiprocessing import Pool

import dotenv
import pandas as pd
from loguru import logger
from modeling.model_class import add_mg_sm
from modeling.model_class_mig import add_mg_sm as add_mg_sm_mig
# from sherpa_ai.policies.react_policy import ReactPolicy
from modeling.react_policy_modeling import ReactPolicy
from modeling.utils import get_llm
from sherpa_ai.agents.qa_agent import QAAgent
from sherpa_ai.events import Event, EventType
from sherpa_ai.memory.belief import Belief
from tqdm import tqdm

dotenv.load_dotenv()


role_description = """You are a helpful agent help user to perform their task.

Your task is to choose from available actions and execute them to help with user's task.

ask clarification if needed.

If you make any assumptions, make sure to clarify them with the user.

You want the response to be concise yet informative and keep the conversation engaging.

Store any relevant clarification into the belief and use it to answer the question if necessary. 

Never Ask question repetitively. Never make random assumption
"""  # noqa E501


def main(configs):
    description, title, args = configs
    # create output folder set the logger to the correct file
    model_name = args.llm
    if "/" in model_name:
        model_name = model_name.split("/")[-1]
    output_folder = (
        f"{args.output_folder}/"
        + model_name
        + f"_output_{args.approach}/run{args.run_number}"
    )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    logger.remove()
    logger.add(
        os.path.join(output_folder, title + ".log"),
        level="INFO",
        colorize=False,
        mode="w",
    )

    belief = Belief()
    llm = get_llm(args.model_type, args.llm)

    policy = ReactPolicy(
        role_description=role_description,
        output_instruction="Determine which action and arguments would be the best continuing the task",  # noqa E501
        llm=llm,
    )

    if args.approach == "sherpa":
        belief = add_mg_sm(belief, llm, output_folder)
    elif args.approach == "sherpa_mig":
        belief = add_mg_sm_mig(belief, llm, output_folder)
    belief.set("description", description)
    belief.set("title", title)
    belief.max_tokens = 1000
    belief.set_current_task(
        Event(EventType.task, "user", "Complete the modeling question")
    )

    # belief.state_machine.enter_state("Waiting")
    qa_agent = QAAgent(llm=llm, belief=belief, num_runs=100, policy=policy)

    belief.state_machine.start()
    while True:
        qa_agent.run()
        if belief.state_machine.get_current_state().name == "end":
            break

    llm_counts = belief.get("num_llm_calls")
    with open(os.path.join(output_folder, title + ".json"), "w") as file:
        json.dump(
            {
                "llm_calls": llm_counts,
            },
            file,
            indent=4,
        )
    # reset the logger
    logger.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform output")
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument(
        "--approach", type=str, default="sherpa", choices=["sherpa", "sherpa_mig"]
    )
    parser.add_argument("--run_number", type=int, required=True)
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()

    # Read the CSV file
    df = pd.read_csv("modeling_problems.csv")

    data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        description = row["description"]
        name = str(index) + "_" + row["name"]
        data.append((description, name, args))
    logger.info(f"Starting generation for {len(data)} problems with {args.llm}")
    with Pool(args.num_processes) as p:
        list(tqdm(p.imap_unordered(main, data), total=len(data)))

    logger.info("Generation completed.")
