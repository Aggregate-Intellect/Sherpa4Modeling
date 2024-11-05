from actions import (CountAll, Filter, GenerateOutput, Output, Query, Related,
                     Same)
from sherpa_ai.actions.base import BaseAction
from sherpa_ai.memory import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine


def get_actions(belief: Belief, llm):
    filter = Filter(belief=belief, name="filter_with_attribute_action")
    query = Query(belief=belief, name="query_attribute_action")
    related = Related(belief=belief, name="get_related_objects_action")
    same = Same(belief=belief, name="get_same_objects_action")
    output = Output(
        belief=belief,
        name="answer_action",
        usage="Return the answer when there is enough information to answer the question",
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


def add_state_machine(belief: Belief, action_map: dict[str, BaseAction]):
    # states = [
    #     {
    #         "name": "Start",
    #         "description": "Given the question, choose the most appropriate type of the final answer of the question is counting (number), querying (a property value) or judging (yes/no question). Choose the appropriate action to go to the corresponding state.",  # noqa
    #     },
    #     "Filtering",
    #     "Checking",
    #     "Relating",
    #     "Querying",
    #     "Finish",
    # ]

    # transitions = [
    #     {
    #         "trigger": "start",
    #         "source": "Start",
    #         "dest": "Filtering",
    #         "before": "count_all_objects_action",
    #     },
    #     {
    #         "trigger": "filter_with_attribute",
    #         "source": "Filtering",
    #         "dest": "Filtering",
    #         "before": "filter_with_attribute_action",
    #     },
    #     {
    #         "trigger": "query",
    #         "source": "Filtering",
    #         "dest": "Querying",
    #     },
    #     {
    #         "trigger": "query_attribute",
    #         "source": "Querying",
    #         "dest": "Querying",
    #         "before": "query_attribute_action",
    #     },
    #     {
    #         "trigger": "relate",
    #         "source": "Querying",
    #         "dest": "Relating",
    #     },
    #     {
    #         "trigger": "get_related_objects",
    #         "source": "Relating",
    #         "dest": "Relating",
    #         "before": "get_related_objects_action",
    #     },
    #     {
    #         "trigger": "same_objects",
    #         "source": "Relating",
    #         "dest": "Checking",
    #     },
    #     {
    #         "trigger": "get_same_objects",
    #         "source": "Checking",
    #         "dest": "Checking",
    #         "before": "get_same_objects_action",
    #     },
    #     {
    #         "trigger": "filter",
    #         "source": "Checking",
    #         "dest": "Filtering",
    #     },
    #     {
    #         "trigger": "answer",
    #         "source": "Filtering",
    #         "dest": "Finish",
    #         "before": "answer_action",
    #     },
    #     {
    #         "trigger": "answer",
    #         "source": "Querying",
    #         "dest": "Finish",
    #         "before": "answer_action",
    #     },
    #     {
    #         "trigger": "answer",
    #         "source": "Relating",
    #         "dest": "Finish",
    #         "before": "answer_action",
    #     },
    #     {
    #         "trigger": "answer",
    #         "source": "Checking",
    #         "dest": "Finish",
    #         "before": "answer_action",
    #     },
    # ]

    states = [
        {
            "name": "Start",
            "description": "Given the question, choose the most appropriate type of the final answer of the question is counting (number), querying (a property value) or judging (yes/no question). Choose the appropriate action to go to the corresponding state.",  # noqa
        },
        "Counting",
        "Querying",
        "Judging",
        "Finish",
    ]

    transitions = [
        {
            "trigger": "count_question",
            "source": "Start",
            "dest": "Counting",
        },
        {
            "trigger": "query_question",
            "source": "Start",
            "dest": "Querying",
        },
        {
            "trigger": "judge_question",
            "source": "Start",
            "dest": "Judging",
        },
        {
            "trigger": "get_related_objects",
            "source": "Judging",
            "dest": "Judging",
            "before": "get_related_objects_action",
        },
        {
            "trigger": "filter_with_attribute",
            "source": "Judging",
            "dest": "Judging",
            "before": "filter_with_attribute_action",
        },
        {
            "trigger": "get_same_objects",
            "source": "Judging",
            "dest": "Judging",
            "before": "get_same_objects_action",
        },
        {
            "trigger": "get_related_objects",
            "source": "Counting",
            "dest": "Counting",
            "before": "get_related_objects_action",
        },
        {
            "trigger": "filter_with_attribute",
            "source": "Counting",
            "dest": "Counting",
            "before": "filter_with_attribute_action",
        },
        {
            "trigger": "get_same_objects",
            "source": "Counting",
            "dest": "Counting",
            "before": "get_same_objects_action",
        },
        {
            "trigger": "answer_counting",
            "source": "Counting",
            "dest": "Finish",
            "before": "answer_count_action",
        },
        {
            "trigger": "answer_querying",
            "source": "Querying",
            "dest": "Finish",
            "before": "answer_querying_action",
        },
        {
            "trigger": "answer_judging",
            "source": "Judging",
            "dest": "Finish",
            "before": "answer_judging_action",
        },
    ]

    sm = SherpaStateMachine(
        states=states,
        transitions=transitions,
        initial="Start",
        action_map=action_map,
        sm_cls=HierarchicalGraphMachine,
    )

    # print(sm.sm.get_graph().draw(None))

    belief.state_machine = sm

    return belief
