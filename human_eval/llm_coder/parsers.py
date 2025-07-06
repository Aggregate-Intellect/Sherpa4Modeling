from langchain_core.output_parsers import BaseOutputParser
import re
from loguru import logger
from pydantic import ConfigDict
from typing import Optional


class PythonProgramParser(BaseOutputParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # The DOTALL flag allows . to match new line characters as well
    pattern: re.Pattern = re.compile("```python(.+)```", flags=re.DOTALL)
    assertion_pattern: re.Pattern = re.compile(r'(?m)^\s*assert\b.*$', re.DOTALL)

    def parse(self, result: str) -> Optional[str]:
        match = self.pattern.search(result)

        if match:
            python_text = match.group(1)
            # Remove all assertions from the code
            python_text = self.assertion_pattern.sub("", python_text)
            return python_text
        else:
            logger.warning(
                f"No python block in the response. The original response is \n {result}")
            return None
        
class PythonOutputParser(BaseOutputParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # The DOTALL flag allows . to match new line characters as well
    pattern: re.Pattern = re.compile("```python(.+)```", flags=re.DOTALL)

    def parse(self, result: str) -> Optional[str]:
        match = self.pattern.search(result)

        if match:
            python_text = match.group(1)
            return python_text
        else:
            logger.warning(
                f"No python block in the response. The original response is \n {result}")
            return None


class TestCaseParser(BaseOutputParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pattern: re.Pattern = re.compile("assert .+")

    def parse(self, result: Optional[str]) -> Optional[list[str]]:
        if result is None:
            return None

        return self.pattern.findall(result)


class PromptExampleParser(BaseOutputParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    patterns: list[re.Pattern] = [
        re.compile(r"^[ \*>]+(.*\(.*) =?[=\-|➞]>? (.+)"),
        re.compile(r"(.+\(.+).+ returns (.+)"),
        re.compile(r">>> (.+)\n +(.+)"),
    ]

    def parse(self, result: str) -> list[dict]:
        matches = []

        for pattern in self.patterns:
            matches = pattern.findall(result)

            if len(matches) > 0:
                break

        if len(matches) == 0:
            logger.warning(
                f"No example in the response. The original response is \n {result}")
            return []
        else:
            return [{"input": match[0].strip(), "output": match[1].strip()} for match in matches]
