import json
import re
from typing import Optional

from langchain_core.prompts import PromptTemplate

from clevr_qa.question_answerer.base import QuestionAnswerer

PROMPT_TEMPLATE = """You are a careful assistant helping a visually impaired person to answer questions regarding a scene. The scene is described in JSON format.

Scene: {scene}

First examine the scene and then check the following question. Reason about how this question can be answered before provide the final answer. The answer should either be a number, yes or no or an attribute value.

Give the final answer in the following JSON format:
```json
{{
    "answer": "<answer to the question>"
}}
```

Question: {question}
"""  # noqa: E501


class DirectAnswerer(QuestionAnswerer):
    prompt_template: PromptTemplate = PromptTemplate.from_template(PROMPT_TEMPLATE)

    def answer_question(self, question: str, scene: dict, idx: int = 0) -> str:
        log_filename = f"{self.log_folder}/direct_{self.llm_name}_qa_{idx}.log"
        llm = self.get_llm(log_filename)
        chain = self.prompt_template | llm

        answer = None
        while answer is None:
            try:
                result = chain.invoke(input={"scene": scene, "question": question})
                answer = extract_json(result.content)
            except json.JSONDecodeError:
                print("JSON Decode Error")
                continue
        return answer, 1


def extract_json(text: str) -> Optional[dict]:
    json_pattern = re.compile(r"```json\n((.|\n)*?)\n```")

    match = json_pattern.search(text)
    if match:
        result = json.loads(match.group(1))
        if isinstance(result, dict) and "answer" in result:
            return result["answer"]
    else:
        return None
