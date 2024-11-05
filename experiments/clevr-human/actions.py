from typing import Any

from langchain_core.prompts import PromptTemplate
from sherpa_ai.actions.base import BaseAction


class Filter(BaseAction):
    name: str = "filter_with_attribute"
    usage: str = "Return the all objects in the scene with the given attribute"
    args: dict = {
        "object_ids": "List of object ids to filter, type: list",
        "attribute": "Name of the attribute to filter, type: str",
        "value": "Value to filter the attribute with, type: str",
    }

    def execute(self, object_ids, attribute, value) -> str:
        scene = self.belief.get("scene")

        if len(object_ids) == 0:
            object_ids = list(range(len(scene["objects"])))
        object_ids = set(object_ids)

        results = []
        for obj_id, obj in enumerate(scene["objects"]):
            if obj_id in object_ids:
                if obj[attribute] == value:
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


OUTPUT_PROMPT_TEMPLATE = """Given the following scene:
{scene}
and the question: {question}
{usage}
"""


class GenerateOutput(BaseAction):
    name: str = "output"
    usage: str = ""
    args: dict = {}
    llm: Any

    def execute(self) -> str:
        prompt_template = PromptTemplate.from_template(OUTPUT_PROMPT_TEMPLATE)

        chain = prompt_template | self.llm

        print(
            prompt_template.invoke(
                input={
                    "scene": self.belief.get("scene"),
                    "question": self.belief.current_task.content,
                    "usage": self.usage,
                }
            )
        )

        return chain.invoke(
            input={
                "scene": self.belief.get("scene"),
                "question": self.belief.current_task.content,
                "usage": self.usage,
            }
        ).content
