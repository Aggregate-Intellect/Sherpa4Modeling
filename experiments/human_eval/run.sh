python run_programs.py --llm_family openai --llm_model gpt-4o-mini --temperature 0.01 --output_filename results/gpt-4o-mini/direct_prompt_test.jsonl --method direct_prompt

python run_programs.py --llm_family openai --llm_model gpt-4o-mini --temperature 0.01 --output_filename results/gpt-4o-mini/agent_coder_improved.jsonl --method agent_coder_improved

python run_programs.py --llm_family togetherai --llm_model meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --temperature 0.01 --output_filename results/Meta-Llama-3.1-70BB-Instruct-Turbo/direct_prompt.jsonl --method direct_prompt

python run_programs.py --llm_family togetherai --llm_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --temperature 0.01 --output_filename results/Meta-Llama-3.1-8B-Instruct-Turbo/agent_coder.jsonl --method agent_coder

# Run Qwen/Qwen2.5-Coder-32B-Instruct
python run_programs.py --llm_family togetherai --llm_model Qwen/Qwen2.5-Coder-32B-Instruct --temperature 0.01 --output_filename results/Qwen2.5-Coder-32B-Instruct/direct_prompt.jsonl --method direct_prompt

python run_programs.py --llm_family togetherai --llm_model Qwen/Qwen2.5-Coder-32B-Instruct --temperature 0.01 --output_filename results/Qwen2.5-Coder-32B-Instruct/state_machine.jsonl --method state_machine

python run_programs.py --llm_family togetherai --llm_model Qwen/Qwen2.5-Coder-32B-Instruct --temperature 0.01 --output_filename results/Qwen2.5-Coder-32B-Instruct/agent_coder.jsonl --method agent_coder


# Run Qwen/Qwen2.5-7B-Instruct-Turbo
python run_programs.py --llm_family togetherai --llm_model Qwen/Qwen2.5-72B-Instruct-Turbo --temperature 0.01 --output_filename results/Qwen2.5-72B-Instruct-Turbo/agent_coder.jsonl --method agent_coder

python run_programs.py --llm_family togetherai --llm_model Qwen/Qwen2.5-7B-Instruct-Turbo --temperature 0.01 --output_filename results/Qwen2.5-7B-Instruct-Turbo/state_machine_1.jsonl --method state_machine

# Test Agent coder
python run_programs.py --llm_family openai --llm_model gpt-4o --temperature 0.01 --output_filename results/gpt-4o/agent_coder.jsonl --method agent_coder

python run_programs.py --llm_family togetherai --llm_model meta-llama/Llama-3.3-70B-Instruct-Turbo-Free --temperature 0.01 --output_filename results/Llama-3.3-70B-Instruct-Turbo-Free/direct_prompt.jsonl --method direct_prompt