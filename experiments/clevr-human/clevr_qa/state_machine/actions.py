from typing import Any, Optional

from langchain_core.prompts import PromptTemplate
from loguru import logger
from sherpa_ai.actions.base import BaseAction
import re
import json
from sherpa_ai.actions.base import ActionArgument
from sherpa_ai.actions.exceptions import SherpaActionExecutionException


class Filter(BaseAction):
    name: str = "filter_with_attribute"
    usage: str = "Return the all objects in the scene with the given attribute value (color, size, shape, material)"  # noqa: E501
    args: list = [
        ActionArgument(
            name="attribute_map",
            type="dict",
            description="Name and value of the attributes to filter",
        )
    ]

    def execute(self, attribute_map) -> str:
        if not isinstance(attribute_map, dict):
            raise SherpaActionExecutionException(
                "Attribute map should be a dictionary, the key must be one of the following: color, size, shape, material"  # noqa: E501
            )

        for key in attribute_map.keys():
            if key not in ["color", "size", "shape", "material"]:
                raise SherpaActionExecutionException(
                    "Attribute map should be a dictionary, the key must be one of the following: color, size, shape, material"  # noqa: E501
                )
        scene = self.belief.get("scene")
        object_ids = list(range(len(scene["objects"])))
        object_ids = set(object_ids)

        results = []
        for obj_id, obj in enumerate(scene["objects"]):
            if obj_id in object_ids:
                include = True
                for attribute, value in attribute_map.items():
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
        return f"There are {len(scene['objects'])} object: {[i for i in range(len(scene['objects']))]}"  # noqa: E501


class Related(BaseAction):
    name: str = "get_related_objects"
    usage: str = "Get the objects spatially related to the given object (left, right, front, behind)"  # noqa: E501
    args: list[ActionArgument] = [
        ActionArgument(
            name="object_id",
            type="int",
            description="Object id to find related objects",
        ),
        ActionArgument(
            name="relation",
            type="str",
            description="Spatial relation to find",
        ),
    ]

    def execute(self, object_id, relation) -> str:
        if relation not in ["left", "right", "front", "behind"]:
            raise SherpaActionExecutionException(
                "Relation should be one of the following: left, right, front, behind"
            )

        object_id = int(object_id)
        scene = self.belief.get("scene")
        object_ids = scene["relationships"][relation][object_id]
        return str(object_ids)


class Query(BaseAction):
    name: str = "query_attribute"
    usage: str = "Query the attribute (color, size, shape, material) of the object given the object id"  # noqa: E501
    args: list[ActionArgument] = [
        ActionArgument(
            name="object_id",
            type="int",
            description="Object id to query",
        ),
        ActionArgument(
            name="attribute",
            type="str",
            description="Attribute to query",
        ),
    ]

    def execute(self, object_id, attribute) -> str:
        if attribute not in ["color", "size", "shape", "material"]:
            raise SherpaActionExecutionException(
                "Attribute should be one of the following: color, size, shape, material"
            )

        object_id = int(object_id)
        scene = self.belief.get("scene")
        return scene["objects"][object_id][attribute]


class Same(BaseAction):
    name: str = "get_same_objects"
    usage: str = "Get the objects that have the same attribute (color, size, shape, material) as the given object"  # noqa: E501
    args: list[ActionArgument] = [
        ActionArgument(
            name="object_id",
            type="int",
            description="Object id to compare",
        ),
        ActionArgument(
            name="attribute",
            type="str",
            description="Attribute name to compare",
        ),
    ]

    def execute(self, object_id, attribute) -> str:
        if attribute not in ["color", "size", "shape", "material"]:
            raise SherpaActionExecutionException(
                "Attribute should be one of the following: color, size, shape, material"
            )

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
        "answer": "Answer to the question. The answer should either be a number, yes or no or an attribute value.",  # noqa: E501
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
"""  # noqa: E501

OUTPUT_PROMPT_TEMPLATE_WITH_SCENE = """You are a careful assistant helping a visually impaired person to answer questions regarding a scene.  The scene is described in JSON format:

Scene: {scene}

Below is some detailed information about the scene.

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
"""  # noqa: E501


class GenerateOutput(BaseAction):
    name: str = "output"
    usage: str = "Output answer to the question"
    args: dict = {}
    llm: Any

    def transform_output(self, output_str: str) -> Optional[str]:
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
        else:
            return None

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

        answer = None

        while answer is None:
            result = chain.invoke(input=input).content
            try:
                answer = self.transform_output(result)
            except json.JSONDecodeError:
                logger.error("JSON Decode Error")
                continue
        return answer
