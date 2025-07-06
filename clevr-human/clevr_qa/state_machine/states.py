from sherpa_ai.actions.base import BaseAction
from sherpa_ai.memory import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine

from clevr_qa.state_machine.actions import (CountAll, Filter, GenerateOutput,
                                            Output, Query, Related, Same)


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
    )  # noqa
    count_all = CountAll(belief=belief, name="count_all_objects_action")
    output_count = Output(
        belief=belief,
        name="answer_count_action",
        args={
            "answer": "Count of the objects satisfying the question, return a number"
        },
        usage="Return answer of the question",
        llm=llm,
    )
    output_judging = Output(
        belief=belief,
        name="answer_judging_action",
        args={"answer": "Answer to the judgement question, return yes or no"},
        usage="Return answer of the question",
        # llm=llm
    )

    output_querying = Output(
        belief=belief,
        name="answer_querying_action",
        args={"answer": "Answer to the question, return an attribute value"},
        usage="Return answer of the question",
        llm=llm,
    )

    return {
        filter.name: filter,
        query.name: query,
        related.name: related,
        same.name: same,
        output.name: output,
        count_all.name: count_all,
        output_count.name: output_count,
        output_judging.name: output_judging,
        output_querying.name: output_querying,
    }


def add_state_machine(
    belief: Belief, action_map: dict[str, BaseAction], print_sm: bool = False
):
    states = [
        {
            "name": "Start",
        },
        {
            "name": "Exploring",
            "initial": "Filtering",
            "children": [
                {
                    "name": "Filtering",
                    "description": "Choose the filter_with_attribute action providing the attribute if you think filter the object is the best action. Otherwise choose other_options to see other options. If you think you can answer the original question, choose answer.",  # noqa: E501
                },
                {
                    "name": "Checking",
                    "description": "perform actions on the filtered objects. If you need to start filtering objects again, choose other_option. If you think you can answer the original question, choose answer.",  # noqa: E501
                },
                {
                    "name": "Relating",
                    "description": "Choose the get_related_objects action with an object id and the relation to get all ids of other objects related to the input object. Otherwise choose other_options to see other options. If you think you can answer the original question, choose answer.",  # noqa
                },  # noqa
                {
                    "name": "Querying",
                    "description": "Choose the query_attribute action providing the object id and the attribute to get the value of the given object and attribute. Otherwise choose other_options to see other options. If you think you can answer the original question, choose answer.",  # noqa
                },  # noqa
            ],
        },
        "Finish",
    ]

    transitions = [
        {
            "trigger": "start",
            "source": "Start",
            "dest": "Exploring",
            "before": "count_all_objects_action",
        },
        {
            "trigger": "filter_with_attribute",
            "source": "Exploring_Filtering",
            "dest": "Exploring_Filtering",
            "before": "filter_with_attribute_action",
        },
        {
            "trigger": "other_options",
            "source": "Exploring_Filtering",
            "dest": "Exploring_Relating",
        },
        {
            "trigger": "get_related_objects",
            "source": "Exploring_Relating",
            "dest": "Exploring_Relating",
            "before": "get_related_objects_action",
        },
        {
            "trigger": "other_options",
            "source": "Exploring_Relating",
            "dest": "Exploring_Checking",
        },
        {
            "trigger": "get_same_objects",
            "source": "Exploring_Checking",
            "dest": "Exploring_Checking",
            "before": "get_same_objects_action",
        },
        {
            "trigger": "other_options",
            "source": "Exploring_Checking",
            "dest": "Exploring_Filtering",
        },
        {
            "trigger": "query_attribute",
            "source": "Exploring",
            "dest": "Exploring",
            "before": "query_attribute_action",
        },
        {
            "trigger": "answer",
            "source": "Exploring",
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
