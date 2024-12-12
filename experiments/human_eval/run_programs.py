from human_eval.data import write_jsonl, read_problems
from direct_prompt import get_openai_coder
from tqdm import tqdm
import json
import os


def load_cache(filename: str) -> list[dict]:
    cache_results = []
    if not os.path.exists(filename):
        return cache_results

    with open(filename, "r") as f:
        for line in f:
            cache_results.append(json.loads(line))

    return cache_results


output_filename = "samples.jsonl"
saving_frequency = 5

problems = read_problems()

num_samples_per_task = 200
task_ids = list(problems.keys())
coder = get_openai_coder("gpt-4o-mini")
print(problems[task_ids[0]]["test"])
print(problems[task_ids[0]]["prompt"])
print(problems[task_ids[0]]["entry_point"])
# results = load_cache(output_filename)
# processed_task_ids = [s["task_id"] for s in results]
# task_ids = [id for id in task_ids if id not in processed_task_ids]

# for i, task_id in enumerate(tqdm(task_ids)):
#     result = {
#         "task_id": task_id,
#         "completion": coder.solve_problem(problems[task_id]["prompt"])
#     }
#     results.append(result)

#     if i % saving_frequency == 0:
#         write_jsonl(output_filename, results)
# write_jsonl(output_filename, results)
