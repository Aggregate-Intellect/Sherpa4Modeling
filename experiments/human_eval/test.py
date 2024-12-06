from human_eval.data import write_jsonl, read_problems

problems = read_problems()

num_samples_per_task = 200
task_ids = list(problems.keys())
print(problems[task_ids[0]]["prompt"])

# samples = [
#     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
#     for task_id in problems
#     for _ in range(num_samples_per_task)
# ]
# write_jsonl("samples.jsonl", samples)