import os

from langchain_core.language_models import LanguageModelLike
from loguru import logger
from modeling.actions import (CheckPlayerRolePattern, GenerateFeedback,
                              IdentifyAbstractClasses, IdentifyAttributes,
                              IdentifyClasses, IdentifyEnumerationClasses,
                              IdentifyNouns, IdentifyPlayerRolePattern,
                              IdentifyRelationships, InspectClass,
                              InspectPattern, IntegrateClasses,
                              IntegrateFeedback, Respond, StartQuestion,
                              SummarizePlayerRolePattern, UserHelp)
from sherpa_ai.actions.base import BaseAction
from sherpa_ai.actions.belief_actions import RetrieveBelief, UpdateBelief
from sherpa_ai.memory.belief import Belief
from sherpa_ai.memory.state_machine import SherpaStateMachine
from transitions.extensions import HierarchicalGraphMachine


def get_actions(
    belief: Belief,
    llm: LanguageModelLike,
    output_folder: str,
) -> dict[str, BaseAction]:
    start_question = StartQuestion(
        name="start_question",
        usage="Start the question answering process",
        belief=belief,
    )

    clarify_question = UserHelp(
        name="clarify_question",
        usage="Ask questions to clarify the intention of user",
        belief=belief,
    )
    answer_question = Respond(
        name="answer_question",
        usage="Answer the user's question based on the current context",
        belief=belief,
    )

    identify_nouns = IdentifyNouns(
        name="identify_nouns",
        usage="Identify nouns from text.",
        belief=belief,
        llm=llm,
    )

    identify_nouns = IdentifyNouns(
        name="identify_nouns",
        usage="Identify nouns based on the modeling problem",
        belief=belief,
        llm=llm,
    )

    identify_classes = IdentifyClasses(
        name="identify_classes",
        usage="Identify classes based on the modeling problem",
        belief=belief,
        llm=llm,
    )

    identify_attributes = IdentifyAttributes(
        name="identify_attributes",
        usage="Identify attributes based on the modeling problem and nouns",
        belief=belief,
        llm=llm,
    )

    identify_enumeration_classes = IdentifyEnumerationClasses(
        name="identify_enumeration_classes",
        usage="Identify enumeration classes based on the current modeling problem",
        belief=belief,
        llm=llm,
    )

    identify_abstract_classes = IdentifyAbstractClasses(
        name="identify_abstract_classes",
        usage="Identify abstract classes based on the current modeling problem",
        belief=belief,
        llm=llm,
    )

    identify_player_role_pattern = IdentifyPlayerRolePattern(
        name="identify_player_role_pattern",
        usage="Identify player role pattern based on the current modeling problem",
        belief=belief,
        llm=llm,
    )

    summarize_player_role_pattern = SummarizePlayerRolePattern(
        name="summarize_player_role_pattern",
        usage="Summarize player role pattern based on the current modeling problem",
        belief=belief,
        llm=llm,
    )

    integrate_classes = IntegrateClasses(
        name="integrate_classes",
        usage="Integrate player role pattern into classes based on the current modeling problem",  # noqa E501
        belief=belief,
        llm=llm,
    )

    generate_feedback = GenerateFeedback(
        name="generate_feedback",
        usage="Generate feeback bsed on the current domain model",
        belief=belief,
        llm=llm,
    )

    integrate_feedback = IntegrateFeedback(
        name="integrate_feedback",
        usage="Integrate feeback into the current domain model",
        belief=belief,
        llm=llm,
    )

    identify_relationships = IdentifyRelationships(
        name="identify_relationships",
        usage="Identify relationships based on the current domain model",
        belief=belief,
        llm=llm,
    )

    inspect_class = InspectClass(
        name="inspect_class",
        usage="Examine if model classes are sufficient based on the current domain model",  # noqa E501
        belief=belief,
        llm=llm,
    )

    inspect_pattern = InspectPattern(
        name="inspect_pattern",
        usage="Examine if a pattern appear in the model",
        belief=belief,
        llm=llm,
    )

    check_pattern = CheckPlayerRolePattern(
        name="check_pattern",
        usage="Examine if there is need of a pattern in the problem description",
        belief=belief,
        llm=llm,
    )

    update_belief = UpdateBelief(belief=belief)

    retrieve_belief = RetrieveBelief(belief=belief)

    def has_player_role_pattern():
        logger.info("has_player_role_pattern", belief.get("check_pattern", None))
        return "Result: True" in belief.get("check_pattern", None)

    def need_improve_class():
        return not ("Result: True" in belief.get("inspect_class", None))

    def need_improve_pattern():
        # if there is a need of pattern, but there is not pattern in the current model,
        # return True.
        # the system needs to regereate pattern
        return "Result: True" in belief.get(
            "check_pattern", None
        ) and "Result: False" in belief.get("inspect_pattern", None)

    def output_model():

        model = belief.get("complete_model")
        logger.info("=" * 40)
        logger.info("end state")

        for e in model:
            logger.info(e)
        logger.info("=" * 40)

        # Define the output file path
        name = belief.get("title")
        output_file_path = os.path.join(output_folder, name + ".txt")

        # Generate the output text
        output_text = "=" * 40 + "\n"
        for e in model:
            output_text += f"{e}\n"
        output_text += "=" * 40 + "\n"

        # Write the output to the file
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(output_text)

        logger.info(f"Output saved to {output_file_path}")

    actions = [
        start_question,
        clarify_question,
        answer_question,
        update_belief,
        retrieve_belief,
        identify_nouns,
        identify_classes,
        identify_enumeration_classes,
        identify_attributes,
        identify_abstract_classes,
        identify_player_role_pattern,
        summarize_player_role_pattern,
        integrate_classes,
        generate_feedback,
        integrate_feedback,
        identify_relationships,
        inspect_class,
        inspect_pattern,
        check_pattern,
    ]

    actions_dict = {action.name: action for action in actions}
    actions_dict["can_generate_feedback"] = generate_feedback.can_execute
    actions_dict["has_player_role_pattern"] = has_player_role_pattern
    actions_dict["need_improve_class"] = need_improve_class
    actions_dict["need_improve_pattern"] = need_improve_pattern
    actions_dict["output_model"] = output_model
    return actions_dict


problem_description = """Hotel Booking Management System (HBMS) Business travellers use HMBS for booking special accommodation deals offered by participating hotels. Travellers register to HBMS by providing their name, billing information (incl. company name and address) and optional travel preferences (e.g. breakfast included, free wifi, 24/7 front desk, etc.). When searching for accommodation, the traveller specifies the city, the date of arrival and departure, the number of needed rooms and the type of rooms (e.g. single, double, twin), minimum hotel rating (stars), a tentative budget (max. cost per night), and optionally, further travel preferences to filter offers in the search results. HBMS lists all available offers of hotels for the given travel period, and the traveller can either create a preliminary booking or complete a booking in the regular way. 
In case of a preliminary booking, HBMS forwards the key parameters of the booking information (i.e. price, city area, hotel rating and key preferences and a unique booking identifier) to other hotels so that they can compete for the traveller with special offers provided within the next 24 hours. After 24-hour deadline, HBMS sends the five best special offers to the traveller who can switch to the new offer or proceed with the original preliminary booking. In both cases, the traveller needs to provide credit card information to finalize a booking. Each finalized booking can be either pre-paid (i.e. paid immediately when it cannot be reimbursed), or paid at hotel (when the traveller pays during his/her stay). A finalized booking needs to be confirmed by the hotel within 24 hours. A booking may also contain a cancellation deadline: if the traveller cancels a confirmed booking before this deadline, then there are no further consequences. However, if a confirmed booking is cancelled after this deadline, then 1-night accommodation is charged for the traveller. HBMS stores all past booking information for a traveller to calculate a reliability rating. Each hotel is located in a city at a particular address, and possibly run by a hotel chain. A hotel may announce its available types of rooms for a given period in HBMS, and may also inform HBMS when a particular type of room is fully booked. HBMS sends information about the preliminary booking information to competitor hotels together with the traveller’s preferences and his/her reliability rating. The competitor hotels may then provide a special offer. Once a booking is finalized, the hotel needs to send a confirmation to the traveller. If a completed booking is not confirmed by the hotel within 24 hours, then HBMS needs to cancel it, and reimburse the traveller in case of a pre-paid booking. If the hotel needs to cancel a confirmed booking, then financial compensation must be offered to the traveller.
"""  # noqa E501

task_description = """
You are a domain modeling expert and are assigned with the task of domain modeling creation.
You objective is to create a textual based domain modeling given the program description.
There are steps involved in the process. Follow the instruction for your current step.
"""  # noqa E501


def add_mg_sm(
    belief: Belief,
    llm: LanguageModelLike,
    output_folder: str,
) -> Belief:
    # Hierarchical version of the state machine
    belief.set("description", problem_description)
    belief.set("task_description", task_description)

    states = [
        "Start",
        {
            "name": "ClassIdentificationState",
            "children": [
                "NounIdentification",
                "ClassIdentification",
                "AttributeIdentification",
                "EnumerationIdentification",
                "AbstractClassIdentification",
            ],
            "initial": "NounIdentification",
        },
        {
            "name": "PlayerRolePatternIdentificationState",
            "children": [
                {
                    "name": "PatternIdentification",
                    "on_enter": "check_pattern",
                },
                "PatternSummarization",
                "PatternIntegration",
            ],
            "initial": "PatternIdentification",
        },
        {
            "name": "FeedbackGenerationState",
            "children": ["FeedbackGeneration", "FeedbackIntegration"],
            "initial": "FeedbackGeneration",
        },
        {"name": "RelationshipIdentificationState"},
        {
            "name": "Inspection",
            "children": [
                {
                    "name": "InspectClass",
                    "on_enter": "inspect_class",
                },
                {
                    "name": "InspectPattern",
                    "on_enter": "inspect_pattern",
                },
            ],
            "initial": "InspectClass",
        },
        {"name": "end", "on_enter": "output_model"},
    ]
    initial = "Start"

    transitions = [
        {
            "trigger": "start",
            "source": "Start",
            "dest": "ClassIdentificationState",
        },
        {
            "trigger": "Identify_nouns",
            "source": "ClassIdentificationState_NounIdentification",
            "dest": "ClassIdentificationState_ClassIdentification",
            "before": "identify_nouns",
        },
        # {
        #     "trigger": "Identify_nouns_again",
        #     "source": "ClassIdentificationState_ClassIdentification",
        #     "dest": "ClassIdentificationState_NounIdentification",
        # },
        {
            "trigger": "Identify_classes",
            "source": "ClassIdentificationState_ClassIdentification",
            "dest": "ClassIdentificationState_AttributeIdentification",
            "before": "identify_classes",
        },
        # {
        #     "trigger": "Identify_classes_again",
        #     "source": "ClassIdentificationState_AttributeIdentification",
        #     "dest": "ClassIdentificationState_ClassIdentification",
        #     "before": "identify_classes",
        # },
        {
            "trigger": "Identify_attributes",
            "source": "ClassIdentificationState_AttributeIdentification",
            "dest": "ClassIdentificationState_EnumerationIdentification",
            "before": "identify_attributes",
        },
        # {
        #     "trigger": "Identify_attributes_again",
        #     "source": "ClassIdentificationState_EnumerationIdentification",
        #     "dest": "ClassIdentificationState_AttributeIdentification",
        # },
        {
            "trigger": "Identify_enumerations",
            "source": "ClassIdentificationState_EnumerationIdentification",
            "dest": "ClassIdentificationState_AbstractClassIdentification",
            "before": "identify_enumeration_classes",
        },
        # {
        #     "trigger": "Identify_enumerations_again",
        #     "source": "ClassIdentificationState_AbstractClassIdentification",
        #     "dest": "ClassIdentificationState_EnumerationIdentification",
        # },
        {
            "trigger": "Identify_abstract_classes",
            "source": "ClassIdentificationState_AbstractClassIdentification",
            "dest": "PlayerRolePatternIdentificationState",
            "before": "identify_abstract_classes",
        },
        # {
        #     "trigger": "Identify_abstract_classes_again",
        #     "source": "PlayerRolePatternIdentificationState",
        #     "dest": "ClassIdentificationState_AbstractClassIdentification",
        # },
        {
            "trigger": "Identify_pattern",
            "source": "PlayerRolePatternIdentificationState_PatternIdentification",
            "dest": "PlayerRolePatternIdentificationState_PatternSummarization",
            "before": "identify_player_role_pattern",
            "conditions": "has_player_role_pattern",
        },
        {
            "trigger": "Generate_feedback",
            "source": "PlayerRolePatternIdentificationState_PatternIdentification",
            "dest": "FeedbackGenerationState",
        },
        {
            "trigger": "Summarize_pattern",
            "source": "PlayerRolePatternIdentificationState_PatternSummarization",
            "dest": "PlayerRolePatternIdentificationState_PatternIntegration",
            "before": "summarize_player_role_pattern",
        },
        {
            "trigger": "Integrate_pattern",
            "source": "PlayerRolePatternIdentificationState_PatternIntegration",
            "dest": "FeedbackGenerationState",
            "before": "integrate_classes",
        },
        {
            "trigger": "Generate_feedback",
            "source": "FeedbackGenerationState_FeedbackGeneration",
            "dest": "FeedbackGenerationState_FeedbackIntegration",
            "before": "generate_feedback",
        },
        {
            "trigger": "Integrate_feedback",
            "source": "FeedbackGenerationState_FeedbackIntegration",
            "dest": "Inspection",
            "before": "integrate_feedback",
        },
        {
            "trigger": "Regenerate_class",
            "source": "Inspection_InspectClass",
            "dest": "ClassIdentificationState_ClassIdentification",
            "conditions": "need_improve_class",
        },
        {
            "trigger": "Inspect_pattern",
            "source": "Inspection_InspectClass",
            "dest": "Inspection_InspectPattern",
        },
        {
            "trigger": "Regenerate_pattern",
            "source": "Inspection_InspectPattern",
            "dest": "PlayerRolePatternIdentificationState",
            "conditions": "need_improve_pattern",
        },
        {
            "trigger": "Finish",
            "source": "Inspection_InspectPattern",
            "dest": "end",
        },
    ]

    action_map = get_actions(belief, llm, output_folder)

    sm = SherpaStateMachine(
        states=states,
        transitions=transitions,
        initial=initial,
        action_map=action_map,
        sm_cls=HierarchicalGraphMachine,
    )

    logger.info(sm.sm.get_graph().draw(None))

    belief.state_machine = sm

    return belief
