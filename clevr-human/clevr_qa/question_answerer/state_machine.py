from sherpa_ai.agents.qa_agent import QAAgent
from sherpa_ai.events import Event, EventType
from sherpa_ai.memory import Belief

from clevr_qa.question_answerer.base import QuestionAnswerer
from clevr_qa.state_machine.policy import ReactPolicy
from clevr_qa.state_machine.states import add_state_machine, get_actions

AGENT_DESCRIPTION = """
You are a question answering assistant helping users to find answers to their questions based on a specific scene.
Each object in the scene contain the following properties: color, size, shape, material, and a unique identifier.
The properties are from a fixed set of values:
– Size: One of large or small.
– Color: One of gray, red, blue, green, brown, purple, cyan, or yellow.
– Shape: One of cube (block), sphere, or cylinder.
– Material: One of rubber (matte) or metal (shinning).
- Unique identifier: The index of the object in the scene, starting from 0.

{scene}

Objects in the scene also have the following relationships: left, right, front or behind.

Given the question, first identify ALL relevant objects in the scene using filter. Then identify their relations.

If answering the question requires and object that does not exist in the scene, give answer "no" if it is a boolean question, or "0" if it is count question.

When provide action arguments, ONLY use the values from the fixed set of values above.
"""  # noqa: E501


class StateMachineAnswerer(QuestionAnswerer):
    use_scene: bool = True
    """Whether the answerer uses the scene information in the prompt"""
    num_runs: int = 10
    """Number of hops when running the state machine"""

    def answer_question(self, question: str, scene: dict, idx: int = 0) -> str:
        log_file = f"{self.log_folder}/state_machine_{self.llm_name}_qa_{idx}.log"
        llm = self.get_llm(log_file)

        belief = Belief()
        action_map = get_actions(belief, llm)
        add_state_machine(belief, action_map, print_sm=False)

        belief.set("scene", scene)

        if self.use_scene:
            belief.set("agent_scene", scene)
            agent_description = AGENT_DESCRIPTION.format(scene=f"Scene: {scene}")
        else:
            agent_description = AGENT_DESCRIPTION.format(scene="")

        policy = ReactPolicy(
            role_description=agent_description,
            llm=llm,
        )

        agent = QAAgent(
            llm=llm,
            belief=belief,
            description=agent_description,
            num_runs=self.num_runs,
            policy=policy,
        )

        belief.set_current_task(Event(EventType.task, "user", f"{question}."))
        agent.run()

        if belief.state_machine.state != "Finish":
            belief.state_machine.answer()
        answer = belief.get("answer_action", "No answer found.")

        return str(answer["answer"]), llm.call_count
