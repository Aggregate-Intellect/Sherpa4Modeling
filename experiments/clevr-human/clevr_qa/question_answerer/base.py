from pydantic import BaseModel
from abc import ABC, abstractmethod
from clevr_qa.utils import get_llm
from langchain_core.language_models import BaseChatModel


class QuestionAnswerer(BaseModel, ABC):
    llm_type: str
    llm_name: str
    temperature: float
    log_folder: str

    @abstractmethod
    def answer_question(self, question: str, scene: dict, idx: int = 0) -> str:
        pass

    def get_llm(self, log_file: str = "") -> BaseChatModel:
        return get_llm(
            self.llm_type,
            self.llm_name,
            self.temperature,
            log_file=log_file,
        )
