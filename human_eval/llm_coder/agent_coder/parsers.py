from langchain_core.output_parsers import BaseOutputParser
from loguru import logger

class PythonProgramParser(BaseOutputParser):
    def parse(self, completion_string: str) -> str:
        if f"```python" in completion_string:
            completion_string = completion_string[completion_string.find(
                f"```python")+len(f"```python"):]
            completion_string = completion_string[:completion_string.find(
                "```")]
        else:
            logger.warning("Error: No code block found")
        return completion_string


class PythonTestParser(BaseOutputParser):
    def parse(self, test_case_string: str) -> str:
        if f"```python" in test_case_string:
            test_case_string = test_case_string[test_case_string.find(f"```python")+len(f"```python"):]
            test_case_string = test_case_string[:test_case_string.find("```")]

        return test_case_string