# State Machine for Clevr Human
Designing a state machine for solving the question answering ask in the Clevr dataset.
1. Install sherpa following the top-level read me

## Create Dataset
1. download the Download CLEVR v1.0 (no images) from [the Clevr website](https://cs.stanford.edu/people/jcjohns/clevr/)
2. put the `CLEVR_val_scenes.json` file to the `data` folder
3. Download human created questions from [here](https://cs.stanford.edu/people/jcjohns/iep/)
4. Put the `CLEVR-Humans-val` file to the `data` folder
5. Run `scripts/create_dataset.py` to create the dataset
7. This script will push the dataset to HuggingFace. Update the `--dataset_name` argument when running the experiments to your dataset name 

6. The processed dataset is also available on [huggingface](https://huggingface.co/datasets/Dogdays/clevr_subset) (Link is mask for double-blind)


## Run Question Answering
* Run the direct prompting approach (Using chain-of-thought)
```
python -m scripts.run_qa --approach direct --num_processes 16 --llm_type openai --llm_name gpt-4o-mini --log_folder logs/direct --output_file direct_prompt_results_gpt-4o-mini.csv
```

* Run the state machine approach
```
python -m scripts.run_qa --approach state_machine --num_processes 16 --llm_type openai --llm_name gpt-4o-mini --log_folder logs/state_machine --output_file state_machine_results_gpt-4o-mini.csv
```

More running command samples are available in the `run.sh` file.