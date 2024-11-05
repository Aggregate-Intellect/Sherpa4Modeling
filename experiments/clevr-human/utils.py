import json


def load_dataset(
    data_folder="data",
    scene_file="CLEVR_val_scenes.json",
    question_file="CLEVR-Humans-val.json",
):
    with open(f"{data_folder}/{scene_file}", "r") as f:
        scenes = json.load(f)["scenes"]

    with open(f"{data_folder}/{question_file}", "r") as f:
        questions = json.load(f)["questions"]

    return scenes, questions


def load_processed_dataset(
    data_folder="data",
    scene_file="CLEVR_val_scenes.json",
    question_file="filtered_questions.jsonl",
):
    with open(f"{data_folder}/{scene_file}", "r") as f:
        scenes = json.load(f)["scenes"]

    with open(f"{data_folder}/{question_file}", "r") as f:
        questions = [json.loads(line) for line in f]

    return scenes, questions
