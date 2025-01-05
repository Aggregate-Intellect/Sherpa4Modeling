from sherpa_ai.memory import Belief
from state_machine.states import add_state_machine
from state_machine.prompts import AGENT_DESCRIPTION_PROMPT
from human_eval.data import write_jsonl, read_problems
from langchain_openai import ChatOpenAI
from sherpa_ai.agents.qa_agent import QAAgent
from sherpa_ai.events import Event, EventType
from sherpa_ai.policies.react_sm_policy import ReactStateMachinePolicy
from tqdm import tqdm
import os
import json

TOGETHER_AI_BASE_URL = "https://api.together.xyz/v1"


def load_cache(filename: str) -> list[dict]:
    cache_results = []
    if not os.path.exists(filename):
        return cache_results

    with open(filename, "r") as f:
        for line in f:
            cache_results.append(json.loads(line))

    return cache_results


def generate_one_task(llm, sample):
    belief = Belief()
    add_state_machine(belief, llm, False)

    belief.set_current_task(
        Event(
            EventType.task, "user", f"Generate a solution for the programming problem: \n{
                sample['prompt']}"
        )
    )
    belief.set("problem", sample)

    policy = ReactStateMachinePolicy(
        llm=llm, role_description=AGENT_DESCRIPTION_PROMPT, output_instruction="")
    qa_agent = QAAgent(
        belief=belief,
        llm=llm,
        descriptoin=AGENT_DESCRIPTION_PROMPT,
        num_runs=100,
        policy=policy
    )

    qa_agent.run()

    return belief.get("generated_solution")


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


def generate_programs(args):
    problems = read_problems()
    task_ids = list(problems.keys())

    llm = get_llm(args.llm_family, args.llm_model, args.temperature)
    results = load_cache(args.output_filename)
    processed_task_ids = [s["task_id"] for s in results]
    task_ids = [id for id in task_ids if id not in processed_task_ids]

    for i, task_id in enumerate(tqdm(task_ids)):
        result = {
            "task_id": task_id,
            "completion": generate_one_task(llm, problems[task_id])
        }
        results.append(result)

        if i % args.saving_frequency == 0:
            write_jsonl(args.output_filename, results)

    write_jsonl(args.output_filename, results)



