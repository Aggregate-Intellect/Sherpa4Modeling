from typing import Any, Optional
from langchain_core.language_models import BaseChatModel
from loguru import logger
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatResult
from langchain_core.messages import BaseMessage, convert_to_openai_messages


class LoggedLLM(BaseChatModel):
    llm: BaseChatModel
    logger_file: str

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        logger_idx = logger.add(self.logger_file)
        context_logger = logger.bind()
        context_logger.debug("***************LLM Input***************")
        for message in convert_to_openai_messages(messages):
            context_logger.debug(f"{message['role']}: {message['content']}")

        result = self.llm._generate(messages, stop, run_manager, **kwargs)

        context_logger.debug("***************LLM Output***************")
        message = convert_to_openai_messages(result.generations[0].message)
        context_logger.debug(f"{message['role']}: {message['content']}")

        logger.remove(logger_idx)
        return result

    def _llm_type(self) -> str:
        return self.llm._llm_type()
