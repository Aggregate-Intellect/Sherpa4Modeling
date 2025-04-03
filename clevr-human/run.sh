python -m scripts.run_qa --approach state_machine --num_processes 16 --llm_type openai --llm_name gpt-4o --log_folder logs/state_mathine/gpt-4o --use_scene --num_runs 10 --output_folder results_new/gpt-4o/run1

python -m scripts.run_qa --approach state_machine --num_processes 8 --llm_type together --llm_name Qwen/Qwen2.5-72B-Instruct-Turbo --log_folder logs/state_machine/Qwen2.5-72B-Instruct-Turbo --use_scene --num_runs 10 --output_folder results_new/Qwen2.5-72B-Instruct-Turbo/run1

python -m scripts.run_qa --approach state_machine --num_processes 8 --llm_type together --llm_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --log_folder logs/direct/Meta-Llama-3.1-70B-Instruct-Turbo --output_folder results_new/Meta-Llama-3.1-70B-Instruct-Turbo/run1


python -m scripts.run_qa --approach routing --num_processes 8 --llm_type together --llm_name Qwen/Qwen2.5-7B-Instruct-Turbo --log_folder logs/routing/Qwen2.5-7B-Instruct-Turbo --use_scene --output_folder results_new/Qwen2.5-7B-Instruct-Turbo/run3
python -m scripts.run_qa --approach routing --num_processes 8 --llm_type openai --llm_name gpt-4o --log_folder logs/routing/gpt-4o --use_scene --output_folder results_new/gpt-4o/run1

python -m scripts.run_qa --approach routing --num_processes 8 --llm_type together --llm_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --log_folder logs/routing/Meta-Llama-3.1-70B-Instruct-Turbo --use_scene --output_folder results_new/Meta-Llama-3.1-70B-Instruct-Turbo/run1


python -m scripts.run_qa --approach react --num_processes 16 --llm_type openai --llm_name gpt-4o --log_folder logs/react/gpt-4o --use_scene --num_runs 10 --output_folder results_new/gpt-4o/run1

python -m scripts.run_qa --approach react --num_processes 8 --llm_type together --llm_name Qwen/Qwen2.5-7B-Instruct-Turbo --log_folder logs/react/Qwen2.5-7B-Instruct-Turbo --use_scene --output_folder results_new/Qwen2.5-7B-Instruct-Turbo/run1

python -m scripts.run_qa --approach react --num_processes 8 --llm_type together --llm_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --log_folder logs/react/Meta-Llama-3.1-70B-Instruct-Turbo --use_scene --output_folder results_new/Meta-Llama-3.1-70B-Instruct-Turbo/run1
