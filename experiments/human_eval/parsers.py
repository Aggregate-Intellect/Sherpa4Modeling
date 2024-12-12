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
    pattern: re.Pattern = re.compile(">>> (.+)\n\s+(.+)")

    def parse(self, result: str) -> list[dict]:
        matches = self.pattern.findall(result)

        if len(matches) == 0:
            logger.warning(
                f"No example in the response. The original response is \n {result}")
            return None
        else:
            return [{"input": match[0], "output": match[1]} for match in matches]
