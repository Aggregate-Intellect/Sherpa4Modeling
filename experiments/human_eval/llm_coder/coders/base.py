from pydantic import BaseModel
from abc import ABC, abstractmethod


class BaseCoder(BaseModel, ABC):
    @abstractmethod
    def solve_problem(self, problem: dict) -> dict:
        pass
