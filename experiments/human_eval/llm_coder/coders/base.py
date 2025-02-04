from pydantic import BaseModel
from abc import ABC, abstractmethod


class CoderResult(BaseModel):
    result: str
    num_llm_calls: int

class BaseCoder(BaseModel, ABC):
    @abstractmethod
    def solve_problem(self, problem: dict) -> CoderResult:
        pass
