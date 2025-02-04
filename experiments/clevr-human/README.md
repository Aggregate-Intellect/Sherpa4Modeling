# State Machine for Clevr Human
Designing a state machine for solving the question answering ask in the Clevr dataset.

## Create Dataset
1. download the Download CLEVR v1.0 (no images) from [the Clevr website](https://cs.stanford.edu/people/jcjohns/clevr/)
2. put the `CLEVR_val_scenes.json` file to the `data` folder
3. Download human created questions from [here](https://cs.stanford.edu/people/jcjohns/iep/)
4. Put the `CLEVR-Humans-val` file to the `data` folder
5. Run `scripts/create_dataset.py` to create the dataset
6. The processed dataset is available on [huggingface](https://huggingface.co/datasets/Dogdays/clevr_subset)


## Run Question Answering
* Run the direct prompting approach (Using chain-of-thought)
```
python -m scripts.run_qa --approach direct --num_processes 16 --llm_type openai --llm_name gpt-4o-mini --log_folder logs/direct --output_file direct_prompt_results_gpt-4o-mini.csv
```

* Run the state machine approach
```
python -m scripts.run_qa --approach state_machine --num_processes 16 --llm_type openai --llm_name gpt-4o-mini --log_folder logs/state_machine --output_file state_machine_results_gpt-4o-mini.csv
```


python -m scripts.run_qa --approach direct --num_processes 8 --llm_type togetherai --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --log_folder logs/Meta-Llama-3.1-8B-Instruct-Turbo/direct_1 --output_file results/direct_prompt_Meta-Llama-3.1-8B-Instruct-Turbo_1.csv --temperature 0.2

python -m scripts.run_qa --approach state_machine --num_processes 5 --llm_type togetherai --llm_name meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --log_folder logs/Meta-Llama-3.1-8B-Instruct-Turbo/state_machine_test --output_file results/state_machine_Meta-Llama-3.1-8B-Instruct-Turbo_test.csv --use_scene --temperature 0.2
