from sherpa_ai.actions.base import BaseAction
from sherpa_ai.memory import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine

from clevr_qa.react.actions import (CountAll, Filter, GenerateOutput, Query,
                                    Related, Same)


def get_actions(belief: Belief, llm):
    filter = Filter(belief=belief, name="filter_with_attribute_action")
    query = Query(belief=belief, name="query_attribute_action")
    related = Related(belief=belief, name="get_related_objects_action")
    same = Same(belief=belief, name="get_same_objects_action")
    output = GenerateOutput(
        belief=belief,
        name="answer_action",
        usage="Return the answer when there is enough information to answer the question",  # noqa: E501
        llm=llm,
    )
    count_all = CountAll(belief=belief, name="count_all_objects_action")

    return {
        filter.name: filter,
        query.name: query,
        related.name: related,
        same.name: same,
        output.name: output,
        count_all.name: count_all,
    }


def add_state_machine(
    belief: Belief, action_map: dict[str, BaseAction], print_sm: bool = False
):
    states = [
        "Start",
        "Reasoning",
        "Finish",
    ]

    transitions = [
        {
            "trigger": "start",
            "source": "Start",
            "dest": "Reasoning",
        },
        {
            "trigger": "filter_with_attribute",
            "source": "Reasoning",
            "dest": "Reasoning",
            "before": "filter_with_attribute_action",
        },
        {
            "trigger": "query_attribute",
            "source": "Reasoning",
            "dest": "Reasoning",
            "before": "query_attribute_action",
        },
        {
            "trigger": "get_related_objects",
            "source": "Reasoning",
            "dest": "Reasoning",
            "before": "get_related_objects_action",
        },
        {
            "trigger": "get_same_objects",
            "source": "Reasoning",
            "dest": "Reasoning",
            "before": "get_same_objects_action",
        },
        {
            "trigger": "count_all_objects",
            "source": "Reasoning",
            "dest": "Reasoning",
            "before": "count_all_objects_action",
        },
        {
            "trigger": "answer",
            "source": "Reasoning",
            "dest": "Finish",
            "before": "answer_action",
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
