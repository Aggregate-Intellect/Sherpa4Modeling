python run_programs.py --llm_family openai --llm_model gpt-4o-mini --temperature 0.01 --output_filename results/gpt-4o-mini/direct_prompt_test.jsonl --method direct_prompt

python run_programs.py --llm_family openai --llm_model gpt-4o-mini --temperature 0.01 --output_filename results/gpt-4o-mini/state_machine.jsonl --method state_machine

python run_programs.py --llm_family togetherai --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --temperature 0.01 --output_filename results/Meta-Llama-3.1-8B-Instruct-Turbo/direct_prompt.jsonl --method direct_prompt

python run_programs.py --llm_family togetherai --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --temperature 0.01 --output_filename results/Meta-Llama-3.1-8B-Instruct-Turbo/state_machine.jsonl --method state_machine