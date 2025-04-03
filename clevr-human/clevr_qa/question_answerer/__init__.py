from clevr_qa.question_answerer.base import QuestionAnswerer
from clevr_qa.question_answerer.dicrect import DirectAnswerer
from clevr_qa.question_answerer.react import ReactQuestionAnswerer
from clevr_qa.question_answerer.routing import RoutingQuestionAnswer
from clevr_qa.question_answerer.state_machine import StateMachineAnswerer


def get_answerer(args) -> QuestionAnswerer:
    if args.approach == "direct":
        return DirectAnswerer(
            llm_type=args.llm_type,
            llm_name=args.llm_name,
            temperature=args.temperature,
            log_folder=args.log_folder,
        )
    if args.approach == "state_machine":
        return StateMachineAnswerer(
            llm_type=args.llm_type,
            llm_name=args.llm_name,
            temperature=args.temperature,
            log_folder=args.log_folder,
            num_runs=args.num_runs,
            use_scene=args.use_scene,
        )
    if args.approach == "react":
        return ReactQuestionAnswerer(
            llm_type=args.llm_type,
            llm_name=args.llm_name,
            temperature=args.temperature,
            log_folder=args.log_folder,
            num_runs=args.num_runs,
        )
    if args.approach == "routing":
        return RoutingQuestionAnswer(
            llm_type=args.llm_type,
            llm_name=args.llm_name,
            temperature=args.temperature,
            log_folder=args.log_folder,
            num_runs=args.num_runs,
        )
    else:
        raise ValueError(f"Unknown approach: {args.approach}")


__all__ = ["DirectAnswerer", "StateMachineAnswerer", "get_answerer"]
