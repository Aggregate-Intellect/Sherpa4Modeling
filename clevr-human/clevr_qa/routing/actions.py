import json
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from sherpa_ai.actions.base import BaseAction

from clevr_qa.utils import JsonOutputParser

ROUTING_PROMPT = """
You are given a question. Your task is to classify the question based on the type of answer it requires:
- If the answer is a number, label it as "counting".
- If the answer is yes or no, label it as "judging".
- If the answer is an attribute value, label it as "querying".

Return your answer as a JSON object in the following format:
{{"type": "<type of question>"}}

For example:
Question: "How many metallic spheres are there on the left of the green cube?"
Output: {{"type": "counting"}}

Question: "Is the purple thing the same shape as the large gray rubber thing?"
Output: {{"type": "judging"}}

Question: "What color is the sphere on the right of the large gray rubber cube?"
Output: {{"type": "querying"}}

Now, classify the following question:
{question}
"""  # noqa: E501


class RoutingAction(BaseAction):
    name: str = "routing_action"
    usage: str = "Route the question to the appropriate action based on its type"
    llm: Any
    args: dict = {}

    def execute(self) -> str:
        prompt = ChatPromptTemplate(
            messages=[
                {
                    "role": "user",
                    "content": ROUTING_PROMPT,
                }
            ]
        )
        question = self.belief.get("question")
        formatted_prompt = prompt.format(question=question)
        parser = JsonOutputParser()

        result = None
        while result is None:
            try:
                result_raw = self.llm.invoke(formatted_prompt).content
                result_raw = parser.parse(result_raw)["type"]
                if not re.match(r"^(counting|judging|querying)$", result_raw):
                    continue
                result = result_raw
                self.belief.set("answer_action", result)
                self.belief.set("question_type", result)

                return result
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")

    def is_judge(self, **kwargs) -> bool:
        question_type = self.belief.get("question_type")
        return question_type == "judging"

    def is_count(self, **kwargs) -> bool:
        question_type = self.belief.get("question_type")
        return question_type == "counting"

    def is_query(self, **kwargs) -> bool:
        question_type = self.belief.get("question_type")
        return question_type == "querying"


JUDGING_PROMPT = """You are a careful assistant helping a visually impaired person to judge if some object exist in a scene. The scene is described in JSON format.

Scene: {scene}

First examine the scene and then check the following question. Reason about how this question can be answered before provide the final answer.
The answer should either be "yes" or "no".

Give the final answer in the following JSON format:
```json
{{
    "answer": "<answer to the question>"
}}
```

Question: {question}
"""


class AnswerJudgingQuestion(BaseAction):
    name: str = "answer_judging_question"
    usage: str = "Answer a judging question based on the scene"
    llm: Any
    args: dict = {}

    def execute(self) -> str:
        prompt = ChatPromptTemplate(
            messages=[
                {
                    "role": "user",
                    "content": JUDGING_PROMPT,
                }
            ]
        )
        question = self.belief.get("question")
        scene = self.belief.get("scene")
        formatted_prompt = prompt.format(question=question, scene=scene)
        parser = JsonOutputParser()

        result = None
        while result is None:
            try:
                result_raw = self.llm.invoke(formatted_prompt).content
                result_raw = parser.parse(result_raw)["answer"]
                if not re.match(r"^(yes|no)$", result_raw):
                    continue
                result = result_raw
                self.belief.set("answer_action", result)
                logger.info(f"Answering judging question: {question} -> {result}")
                return result
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")


EXTRACTION_PROMPT = """You are a careful assistant helping a visually impaired person to extract some objects satisfying the question description. The scene is described in JSON format.

Scene: {scene}

First examine the scene and then check the following question. Reason about how this question can be answered before provide the final answer.
The answer should be a list of ids of objects in the scene that satisfy the condition in the question, make sure the list is parsable as JSON.
If there is no object satisfying the condition, or the condition cannot be met, return an empty list.

Give the final answer in the following JSON format:
```json
{{
    "objects": "<list of object ids satisfying the condition>"
}}
```

Question: {condition}
"""  # noqa: E501


class ExtractObjectsAction(BaseAction):
    name: str = "extract_objects_action"
    usage: str = "Extract objects satisfying the question based on the scene"
    llm: Any
    args: dict = {}

    def execute(
        self,
    ) -> str:
        prompt = ChatPromptTemplate(
            messages=[{"role": "user", "content": EXTRACTION_PROMPT}]
        )
        question = self.belief.get("question")
        scene = self.belief.get("scene").copy()
        scene["objects"] = add_object_ids(scene["objects"])

        formatted_prompt = prompt.format(condition=question, scene=scene)
        parser = JsonOutputParser()

        result = None
        while result is None:
            try:
                result_raw = self.llm.invoke(formatted_prompt).content
                result_raw = parser.parse(result_raw)["objects"]
                if not isinstance(result_raw, list) and not isinstance(
                    result_raw, int
                ):
                    continue

                if isinstance(result_raw, int):
                    result_raw = [result_raw]
                result = result_raw
                self.belief.set("extracted_objects", result)
                logger.info(f"Extracting objects: {question} -> {result}")
                return result
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")


class AnswerCountingQuestion(BaseAction):
    name: str = "answer_counting_question"
    usage: str = "Answer a counting question based on the scene"
    args: dict = {}

    def execute(self) -> str:
        objects = self.belief.get("extracted_objects")

        self.belief.set("answer_action", len(objects))

        return len(objects)


QUERYING_PROMPT = """
You are a careful assistant helping a visually impaired person to answer a question about some object attribute in the scene. The scene is described in JSON format.

Scene: {scene}

First examine the scene and then check the following question. Reason about how this question can be answered before provide the final answer.
The answer should be a signle valid attribute value, including exactly one of the the following:
- color: gray, red, blue, green, brown, purple, cyan, yellow
- size: large, small
- shape: cube, sphere, cylinder
- material: rubber, metal

Give the final answer in the following JSON format:
```json
{{
    "answer": "<answer to the question>"
}}
```

Question: {question}
"""


class AnswerQueryingQuestion(BaseAction):
    name: str = "answer_querying_question"
    usage: str = "Answer a querying question based on the scene"
    llm: Any
    args: dict = {}

    def execute(self) -> str:
        prompt = ChatPromptTemplate(
            messages=[
                {
                    "role": "user",
                    "content": QUERYING_PROMPT,
                }
            ]
        )
        question = self.belief.get("question")
        scene = self.belief.get("scene")
        parser = JsonOutputParser()

        result = None
        while result is None:
            try:
                formatted_prompt = prompt.format(question=question, scene=scene)
                result_raw = self.llm.invoke(formatted_prompt).content
                result_raw = parser.parse(result_raw)["answer"].lower()
                if not check_attribute_value(result_raw):
                    question = f'{question}\n"{result_raw}" is not a valid attribute value. Please provide a valid attribute value.'
                    continue
                result = result_raw
                self.belief.set("answer_action", result)
                return result
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")


def check_attribute_value(result: str) -> bool:
    valid_values = [
        "gray",
        "red",
        "blue",
        "green",
        "brown",
        "purple",
        "cyan",
        "yellow",
        "large",
        "small",
        "cube",
        "sphere",
        "cylinder",
        "rubber",
        "metal",
    ]

    return result in valid_values


def add_object_ids(objects: list):
    for i, obj in enumerate(objects):
        obj["id"] = i
    return objects
