from langchain_core.output_parsers import BaseOutputParser
import re
from loguru import logger
from pydantic import ConfigDict
from typing import Optional


class PythonOutputParser(BaseOutputParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    # The DOTALL flag allows . to match new line characters as well
    pattern: re.Pattern = re.compile("```python(.+)```", flags=re.DOTALL)

    def parse(self, result: str) -> Optional[str]:
        match = self.pattern.search(result)

        if match:
            return match.group(1)
        else:
            logger.warning(
                f"No python block in the response. The original response is \n {result}")
            return None


class PromptExampleParser(BaseOutputParser):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    patterns: list[re.Pattern] = [
        re.compile("^[ \*>]+(.*\(.*) =?[=\-|➞]>? (.+)"),
        re.compile("(.+\(.+).+ returns (.+)"),
        re.compile(">>> (.+)\n +(.+)"),
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
