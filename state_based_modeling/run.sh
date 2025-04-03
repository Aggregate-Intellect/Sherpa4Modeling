# Or gpt-4o-mini
python direct_prompt.py --model_type=openai --llm=gpt-4o --run_number 3

# Or meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo, Qwen/Qwen2.5-7B-Instruct-Turbo, Qwen/Qwen2.5-72B-Instruct-Turbo
python direct_prompt.py --model_type=together --llm=Qwen/Qwen2.5-72B-Instruct-Turbo --run_number 3


python main_class.py --model_type=openai --llm=gpt-4o-mini --run_number 1 --approach sherpa_mig

python main_class.py --model_type=together --llm=Qwen/Qwen2.5-7B-Instruct-Turbo --run_number 1 --approach sherpa_mig
python main_class.py --model_type=together --llm=meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --run_number 1 --approach sherpa_mig