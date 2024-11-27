from typing import Any

from langchain_core.prompts import PromptTemplate
from sherpa_ai.actions.base import BaseAction
import re
import json


class Filter(BaseAction):
    name: str = "filter_with_attribute"
    usage: str = "Return the all objects in the scene with the given attribute"
    args: dict = {
        # "object_ids": "List of object ids to filter, type: list",
        "attributes": "Name and value of the attributes to filter, type: dict",
    }

    def execute(self, attributes) -> str:
        scene = self.belief.get("scene")

        # if len(object_ids) == 0:
        object_ids = list(range(len(scene["objects"])))
        object_ids = set(object_ids)

        results = []
        for obj_id, obj in enumerate(scene["objects"]):
            if obj_id in object_ids:
                include = True
                for attribute, value in attributes.items():
                    if obj[attribute.lower()] != value.lower():
                        include = False
                        break
                if include:
                    results.append(obj_id)

        return str(results)


class CountAll(BaseAction):
    name: str = "count_all_objects"
    usage: str = "Count all the objects in the scene"
    args: dict = {}

    def execute(self) -> str:
        scene = self.belief.get("scene")
        return f"There are {len(scene['objects'])} object: {[i for i in range(len(scene['objects']))]}"


class Related(BaseAction):
    name: str = "get_related_objects"
    usage: str = "Get the objects spatially related to the given object"
    args: dict = {
        "object_id": "Object id to find related objects, type: int",
        "relation": "Spatial relation to find, type: str",
    }

    def execute(self, object_id, relation) -> str:
        object_id = int(object_id)
        scene = self.belief.get("scene")
        object_ids = scene["relationships"][relation][object_id]
        return str(object_ids)


class Query(BaseAction):
    name: str = "query_attribute"
    usage: str = "Query the attribute of the object given the object id"
    args: dict = {
        "object_id": "Object id to query, type: int",
        "attribute": "Attribute to query, type: str",
    }

    def execute(self, object_id, attribute) -> str:
        object_id = int(object_id)
        scene = self.belief.get("scene")
        return scene["objects"][object_id][attribute]


class Same(BaseAction):
    name: str = "get_same_objects"
    usage: str = "Get the objects that have the same attribute as the given object"
    args: dict = {
        "object_id": "Object id to compare, type: int",
        "attribute": "Attribute name to compare, type: str",
    }

    def execute(self, object_id, attribute) -> str:
        object_id = int(object_id)
        scene = self.belief.get("scene")
        attribute_value = scene["objects"][object_id][attribute]

        object_ids = []

        for obj_id, obj in enumerate(scene["objects"]):
            if obj_id == object_id:
                continue

            if obj[attribute] == attribute_value:
                object_ids.append(obj_id)

        return str(object_ids)


class Output(BaseAction):
    name: str = "output"
    usage: str = "Output answer to the question"
    args: dict = {
        "answer": "Answer to the question. The answer should either be a number, yes or no or an attribute value.",
    }

    def execute(self, answer) -> str:
        return answer


OUTPUT_PROMPT_TEMPLATE = """You are a careful assistant helping a visually impaired person to answer questions regarding a scene. Below is some information about the scene.

Information:
{information}

First examine the scene and then check the following question. Reason about how this question can be answered concisely before provide the final answer. The answer should either be a number, yes or no or an attribute value.

Give the final answer in the following JSON format:
```json
{{
    "answer": "<answer to the question>"
}}
```

Question: {question}
"""

OUTPUT_PROMPT_TEMPLATE_WITH_SCENE = """You are a careful assistant helping a visually impaired person to answer questions regarding a scene.  The scene is described in JSON format:

Scene: {scene}

Below is some information about the scene.

Information:
{information}

First examine the scene and then check the following question. Reason about how this question can be answered concisely before provide the final answer. The answer should either be a number, yes or no or an attribute value.

Give the final answer in the following JSON format:
```json
{{
    "answer": "<answer to the question>"
}}
```

Question: {question}
"""


class GenerateOutput(BaseAction):
    name: str = "output"
    usage: str = "Output answer to the question"
    args: dict = {}
    llm: Any

    def transform_output(self, output_str: str) -> str:
        """
        Transform the output string into an action and arguments

        Args:
            output_str: Output string

        Returns:
            str: Action to be taken
            dict: Arguments for the action
        """
        json_pattern = re.compile(r"(\{.*\})", re.DOTALL)
        match = json_pattern.search(output_str)

        if match is not None:
            return json.loads(match.group(1))

    def execute(self) -> str:
        scene = self.belief.get("agent_scene", None)
        information = self.belief.get_internal_history(lambda _: 0)

        if scene is not None:
            prompt_template = PromptTemplate.from_template(
                OUTPUT_PROMPT_TEMPLATE_WITH_SCENE
            )
            input = {
                "scene": scene,
                "information": information,
                "question": self.belief.current_task.content,
            }
        else:
            prompt_template = PromptTemplate.from_template(OUTPUT_PROMPT_TEMPLATE)
            input = {
                "information": information,
                "question": self.belief.current_task.content,
            }

        chain = prompt_template | self.llm

        result = chain.invoke(input=input).content

        return self.transform_output(result)
