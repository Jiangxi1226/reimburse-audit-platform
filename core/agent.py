from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, name: str, llm, system_prompt: str = ""):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt

    @abstractmethod
    def run(self, user_input: str) -> str:
        ...
