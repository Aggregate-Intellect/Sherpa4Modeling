from sherpa_ai.actions.base import BaseAction
from sherpa_ai.memory import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine
from llm_coder.test_based_sm.actions import GenerateSolution, GenerateTestCases, EvaluateSolution, CheckCorrectness
from langchain_core.language_models import BaseChatModel


def get_actions(belief: Belief, llm: BaseChatModel):
    generate_solution = GenerateSolution(belief=belief, llm=llm)
    generate_test_cases = GenerateTestCases(belief=belief, llm=llm)
    evaluate_solution = EvaluateSolution(belief=belief, llm=llm)
    check_correctness = CheckCorrectness(belief=belief, llm=llm)

    return {
        "generate_solution": generate_solution,
        "generate_test_cases": generate_test_cases,
        "evaluate_solution": evaluate_solution,
        "check_correctness": check_correctness
    }


def add_state_machine(belief: Belief, llm: BaseChatModel, print_sm: bool = False):
    states = [
        "Start",
        "SolutionGenerated",
        "TestCasesGenerated",
        "SolutionEvaluated",
        "Finish"
    ]

    transitions = [
        {
            "trigger": "generate_tests",
            "source": "Start",
            "dest": "TestCasesGenerated",
            "before": "generate_test_cases"
        },
        {
            "trigger": "generate_next_solution",
            "source": "TestCasesGenerated",
            "dest": "SolutionGenerated",
            "before": "generate_solution"
        },
        {
            "trigger": "evaluate_current_solution",
            "source": "SolutionGenerated",
            "dest": "SolutionEvaluated",
            "before": "evaluate_solution"
        },
        {
            "trigger": "finish_generation",
            "source": "SolutionEvaluated",
            "dest": "Finish",
            "conditions": "check_correctness"
        },
        {
            "trigger": "retry_generation",
            "source": "SolutionEvaluated",
            "dest": "TestCasesGenerated",
            "unless": "check_correctness"
        }
    ]

    action_map = get_actions(belief, llm)
    sm = SherpaStateMachine(
        states=states,
        transitions=transitions,
        initial="Start",
        action_map=action_map,
        sm_cls=HierarchicalGraphMachine,
    )

    if print_sm:
        print(sm.get_graph().draw(None))

    belief.state_machine = sm

    return sm
