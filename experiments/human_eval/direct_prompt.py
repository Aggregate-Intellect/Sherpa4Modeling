from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from parsers import PythonOutputParser

PROMPT_TEMPLATE = """
You are a Python programming expert tasked with solving algorithmic problems. Your task is to write a Python function that satisfies the following requirements. Please ensure that your solution:
1. Is written in Python 3.
2. Is aligned with the problem description written in the function comment.
2. Passes all the test cases provided in the problem description.
3. Give a runnable problem including potential imports

First explain how to solve the programming challenge, then output the final Python program as a Markdown code block:
```python
```

## Problem
{problem}

## Solution
"""


class DirectPromptCoder(BaseModel):
    llm: BaseChatModel
    parser: PythonOutputParser = PythonOutputParser()

    def solve_problem(self, problem):
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = prompt | self.llm | self.parser
        result = None
        while result is None:
            result = chain.invoke({
                "problem": problem
            })
        return result


def get_openai_coder(model_name: str, temperature: float = 0.01):
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    coder = DirectPromptCoder(llm=llm)

    return coder
