from clevr_qa.utils import load_dataset
import json
import random
from argparse import ArgumentParser
import pandas as pd
from datasets import Dataset

# The skip indices are constructed by manually going through the randomly shuffled
# questions and finding indices of questions that cannot be answered by just using
# the scene information. For example, questions that involving checking the
# reflection of an object.
# These questions have the following issues (in order):
# wrong answer
# wrong answer
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
COUNT_SKIP_INDICES = [2, 15, 24, 34, 36, 37, 38, 39]
# These questions have the following issues (in order):
# not answerable without image
# wrong answer
# not answerable without image
# the question is a bit ambiguous
# typo in the question
JUDGE_SKIP_INDICES = [1, 3, 4, 5, 34]
# These questions have the following issues (in order):
# not answerable without image
# not answerable without image
# wrong answer
# wrong answer
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
# not answerable without image
# wrong answer
# not answerable without image
# not answerable without image
# not answerable without image
QUERY_SKIP_INDICES = [4, 5, 11, 15, 18, 19, 29, 33, 38, 39, 42, 43, 44, 45, 46]

SKIP_INDICES = {
    "count": COUNT_SKIP_INDICES,
    "judge": JUDGE_SKIP_INDICES,
    "query": QUERY_SKIP_INDICES,
}


def process_scene(scene: dict) -> dict:
    if "image_index" in scene:
        scene.pop("image_index")

    for obj in scene["objects"]:
        if "rotation" in obj:
            obj.pop("rotation")
        if "pixel_coords" in obj:
            obj.pop("pixel_coords")
        if "3d_coords" in obj:
            obj.pop("3d_coords")

    if "image_filename" in scene:
        scene.pop("image_filename")

    if "split" in scene:
        scene.pop("split")

    if "directions" in scene:
        scene.pop("directions")

    return scene


def get_question_by_category(
    questions: list[dict], category: str, count: int = 20
) -> list[dict]:
    filtered_questions = []

    for question in questions:
        if category == "count" and question["answer"].isdigit():
            filtered_questions.append(question)
        elif category == "judge" and question["answer"].lower() in ["yes", "no"]:
            filtered_questions.append(question)
        elif (
            category == "query"
            and question["answer"].lower() not in ["yes", "no"]
            and not question["answer"].isdigit()
        ):
            filtered_questions.append(question)

        if len(filtered_questions) >= count:
            break

    return filtered_questions


def filter_dataset(questions: list[dict]) -> list[dict]:
    categories = ["count", "judge", "query"]
    counts = [33, 33, 34]
    processed_questions = []

    for category in categories:
        filtered_questions = get_question_by_category(questions, category, 50)

        target_count = counts.pop(0)
        current_count = 0
        skip_indices = SKIP_INDICES[category]
        for i, question in enumerate(filtered_questions):
            if i in skip_indices:
                continue
            processed_questions.append(question)
            current_count += 1

            if current_count >= target_count:
                break

    return processed_questions


def main(args):
    random.seed(42)
    scenes, questions = load_dataset()
    random.shuffle(questions)

    processed_questions = filter_dataset(questions)

    samples = []
    for q in processed_questions:
        scene = scenes[q["image_index"]]
        samples.append(
            {
                "question": q["question"],
                "answer": q["answer"],
                "image": q["image_filename"],
                "scene": process_scene(scene),
            }
        )

    with open(args.output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    dataframe = pd.DataFrame(samples)
    dataset = Dataset.from_pandas(dataframe)
    dataset.push_to_hub(args.hg_dataset_name, private=args.hg_set_private)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        default="data/filtered_questions.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--hg_dataset_name",
        type=str,
        required=True,
        help="Name of the dataset to be uploaded to Hugging Face Hub",
    )
    parser.add_argument(
        "--hg_set_private",
        action="store_true",
        help="Set the dataset to private on Hugging Face Hub",
    )
    args = parser.parse_args()

    main(args)
