from llm_coder.coders.base import BaseCoder
from sherpa_ai.memory import Belief
from sherpa_ai.agents.qa_agent import QAAgent
from sherpa_ai.events import Event, EventType
from sherpa_ai.policies.react_sm_policy import ReactStateMachinePolicy
from llm_coder.test_based_sm.prompts import AGENT_DESCRIPTION_PROMPT
from llm_coder.test_based_sm.states import add_state_machine
from langchain_core.language_models import BaseChatModel


class TestBasedSMCoder(BaseCoder):
    llm: BaseChatModel

    def solve_problem(self, problem):
        belief = Belief()
        add_state_machine(belief, self.llm, False)

        belief.set_current_task(
            Event(
                EventType.task, "user", f"Generate a solution for the programming problem: \n{
                    problem['prompt']}"
            )
        )
        belief.set("problem", problem)

        policy = ReactStateMachinePolicy(
            llm=self.llm, role_description=AGENT_DESCRIPTION_PROMPT, output_instruction="")
        qa_agent = QAAgent(
            belief=belief,
            llm=self.llm,
            descriptoin=AGENT_DESCRIPTION_PROMPT,
            num_runs=100,
            policy=policy
        )

        qa_agent.run()

        return belief.get("generated_solution")
