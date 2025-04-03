from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Optional, Tuple

from langchain_core.language_models import BaseLanguageModel
from loguru import logger
from sherpa_ai.policies.base import BasePolicy, PolicyOutput

if TYPE_CHECKING:
    from sherpa_ai.actions.base import BaseAction
    from sherpa_ai.memory.belief import Belief

SELECTION_DESCRIPTION = """{role_description}

**Possible Actions**:
{possible_actions}

**History of Previous Actions**:
{history_of_previous_actions}

You should only select the actions specified in **Possible Actions**

Response Format:
{response_format}
Ensure the response can be parsed by Python json.loads

**Overall Question Description**: {task_description}

**Current Task Description: {state_description}

First, concisely summarize the current available information, then choose the most appropriate action and its corresponding arguments.

"""  # noqa: E501


class ReactPolicy(BasePolicy):
    """
    The policy to select an action from the belief based on the ReACT framework.

    See this paper for more details: https://arxiv.org/abs/2210.03629

    Attributes:
        role_description (str): The description of the agent role to help select an action
        llm (BaseLanguageModel): The large language model used to generate text
        description (str): Description to select the action from the belief
        response_format (dict): The response format for the policy in JSON format
    """  # noqa: E501

    role_description: str
    llm: BaseLanguageModel = None
    description: str = SELECTION_DESCRIPTION
    response_format: dict = {
        "command": {
            "name": "tool/command name you choose",
            "args": {"arg name": "value"},
        },
    }
    checked_up_to: int = 1

    def transform_output(self, output_str: str) -> Tuple[str, dict]:
        """
        Transform the output string into an action and arguments

        Args:
            output_str: Output string

        Returns:
            str: Action to be taken
            dict: Arguments for the action
        """
        json_pattern = re.compile(r"(\{.*\})", re.DOTALL)
        match = json_pattern.search(output_str)

        if match is not None:
            result = match.group(1)
            result.replace("'", '"')
            output = json.loads(match.group(1))
        else:
            logger.error("Output does not contain proper json format {}", output_str)
            return "Finished", None
        command = output["command"]
        name = command["name"]
        args = command.get("args", {})
        return name, args

    def is_selection_trivial(self, actions: list[BaseAction]) -> bool:
        """
        Check if the selection of the action is trivial. The selection is trivial if there
        is only one action without any arguments, so LLM is not needed in the selection.

        Args:
            belief (Belief): The current state of the agent

        Returns:
            bool: True if the selection is trivial, False otherwise
        """  # noqa: E501
        return len(actions) == 1 and len(actions[0].args) == 0

    def select_action(self, belief: Belief) -> Optional[PolicyOutput]:
        """
        Select an action from a list of possible actions based on the current state (belief)

        Args:
            belief (Belief): The current state of the agent

        Returns:
            Optional[PolicyOutput]: The selected action and arguments, or None if the selected
            action is not found in the list of possible actions
        """  # noqa: E501
        # if the last action results in an error, go to the next state
        if len(belief.internal_events) > self.checked_up_to:
            self.checked_up_to = len(belief.internal_events)
            if "error" in belief.internal_events[-1].content.lower():
                logger.error(belief.internal_events[-1].content)
                return PolicyOutput(action=belief.get_action("other_options"), args={})
            if (
                len(belief.internal_events) > 4
                and belief.internal_events[-4].content
                == belief.internal_events[-2].content
            ):
                return PolicyOutput(action=belief.get_action("other_options"), args={})

        actions = belief.get_actions()

        if self.is_selection_trivial(actions):
            return PolicyOutput(action=actions[0], args={})

        task_description = belief.current_task.content
        possible_actions = "\n".join([str(action) for action in actions])
        history_of_previous_actions = belief.get_internal_history(
            self.llm.get_num_tokens
        )

        state_description = belief.state_machine.sm.get_state(
            belief.state_machine.state
        ).description
        if state_description is None:
            state_description = (
                "Choose the most appropriate action to go to the corresponding state."
            )

        response_format = json.dumps(self.response_format, indent=4)

        prompt = self.description.format(
            role_description=self.role_description,
            task_description=task_description,
            possible_actions=possible_actions,
            history_of_previous_actions=history_of_previous_actions,
            state_description=state_description,
            response_format=response_format,
        )
        # logger.info(f"Prompt: {prompt}")
        result = self.llm.predict(prompt)
        # logger.info(f"Result: {result}")
        name, args = self.transform_output(result)

        action = belief.get_action(name)

        if action is None:
            raise ValueError(f"Action {name} not found in the list of possible actions")

        return PolicyOutput(action=action, args=args)
