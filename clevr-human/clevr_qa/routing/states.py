from sherpa_ai.actions.base import BaseAction
from sherpa_ai.memory import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine

from clevr_qa.routing.actions import (AnswerCountingQuestion,
                                      AnswerJudgingQuestion,
                                      AnswerQueryingQuestion,
                                      ExtractObjectsAction, RoutingAction)


def get_actions(belief: Belief, llm):
    routing = RoutingAction(
        name="routing_action",
        belief=belief,
        llm=llm,
    )
    judging_action = AnswerJudgingQuestion(
        name="answering_judging_action",
        belief=belief,
        llm=llm,
    )

    extract_objects_action = ExtractObjectsAction(
        name="extract_objects_action",
        belief=belief,
        llm=llm,
    )

    count_action = AnswerCountingQuestion(
        name="answer_counting_question",
        belief=belief,
    )

    answer_querying_action = AnswerQueryingQuestion(
        name="answer_querying_question",
        belief=belief,
        llm=llm,
    )

    return {
        routing.name: routing,
        "is_judging": routing.is_judge,
        "is_counting": routing.is_count,
        "is_querying": routing.is_query,
        judging_action.name: judging_action,
        extract_objects_action.name: extract_objects_action,
        count_action.name: count_action,
        answer_querying_action.name: answer_querying_action,
    }


def add_state_machine(
    belief: Belief, action_map: dict[str, BaseAction], print_sm: bool = False
):
    states = [
        "Start",
        {
            "name": "Routing",
            "on_enter": "routing_action",
        },
        {
            "name": "Extraction",
            "on_enter": "extract_objects_action",
        },
        "Finish",
    ]

    transitions = [
        {
            "trigger": "start",
            "source": "Start",
            "dest": "Routing",
        },
        {
            "trigger": "answering_judging",
            "source": "Routing",
            "dest": "Finish",
            "before": "answering_judging_action",
            "conditions": "is_judging",
        },
        {
            "trigger": "extraction",
            "source": "Routing",
            "dest": "Extraction",
            "conditions": "is_counting",
        },
        {
            "trigger": "answer_counting",
            "source": "Extraction",
            "dest": "Finish",
            "before": "answer_counting_question",
        },
        {
            "trigger": "answer_query",
            "source": "Routing",
            "dest": "Finish",
            "conditions": "is_querying",
            "before": "answer_querying_question",
        },
    ]

    sm = SherpaStateMachine(
        states=states,
        transitions=transitions,
        initial="Start",
        action_map=action_map,
        sm_cls=HierarchicalGraphMachine,
    )

    if print_sm:
        print(sm.sm.get_graph().draw(None))

    belief.state_machine = sm

    return belief
