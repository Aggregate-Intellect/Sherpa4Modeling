from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine
from sherpa_ai.memory import Belief
from langchain_core.language_models import BaseChatModel
from llm_coder.agent_coder_improved.actions import GenerateProgram, GenerateTest, EvaluateProgram

def get_actions(belief: Belief, llm: BaseChatModel):
    generate_program = GenerateProgram(belief=belief, llm=llm)
    generate_tests = GenerateTest(belief=belief, llm=llm)
    evaluate_program = EvaluateProgram(belief=belief)

    return {
        "generate_program_action": generate_program,
        "generate_tests_action": generate_tests,
        "evaluate_solution_action": evaluate_program,
        "is_passed": evaluate_program.is_passed
    }


def add_state_machine(belief: Belief, llm: BaseChatModel, print_sm: bool = False) -> SherpaStateMachine:
    states = [
        "Start",
        {
            "name": "GenerateProgram",
            "on_enter": "generate_program_action",
        },
        {
            "name": "GenerateTestCases",
            "on_enter": "generate_tests_action",
        },
        {
            "name": "EvaluateSolution",
            "on_enter": "evaluate_solution_action",
        },
        "Finish"
    ]

    transitions = [
        {
            "trigger": "generate_program",
            "source": "Start",
            "dest": "GenerateProgram",
        },
        {
            "trigger": "generate_test_cases",
            "source": "GenerateProgram",
            "dest": "GenerateTestCases",
        },
        {
            "trigger": "evaluate_solution",
            "source": "GenerateTestCases",
            "dest": "EvaluateSolution",
        },
        {
            "trigger": "finish_generation",
            "source": "EvaluateSolution",
            "dest": "Finish",
            "conditions": "is_passed"
        },
        {
            "trigger": "retry_generation",
            "source": "EvaluateSolution",
            "dest": "GenerateProgram",
            "unless": "is_passed"
        }
    ]

    action_map = get_actions(belief, llm)
    sm = SherpaStateMachine(
        states=states,
        transitions=transitions,
        initial="Start",
        action_map=action_map,
        sm_cls=HierarchicalGraphMachine
    )

    if print_sm:
        print(sm.get_graph().draw(None))

    belief.state_machine = sm

    return sm
