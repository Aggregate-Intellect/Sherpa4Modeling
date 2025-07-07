# Sherpa for Clevr Human (The Question Answering Use Case)

Designing a state machine for solving the question answering ask in the Clevr-Human dataset.

## Organization
The use case is organized as follows:
* `clevr_qa` contains the code implementation for the use case
   * The folder `react` contains the implementation of the ReACT approach
   * The folder `routing` contains the implementation of the routing state machine approach
   * The folder `state_machine` contains the implementation of the planning state machine approach
   * The `question_answering/direct.py` file contains the implementation of the direct approach
* `data` contains the processed dataset for this use case
* `results_new` contains the cached results of the experiments run on the use case
* `scripts` contains the scripts to create the dataset and run the question answering task
* `.env_template` file contains the environment variables required to run the use case.
* `requirements.txt` contains the requirements to run the use case
* `evaluation.ipynb` contains the evaluation notebook to create tables and figures in the paper for the Clevr-Human use case

## Installation

1. Create a new virtual environment for this use case (not required, but recommended)

   ```bash
   # For venv
   python -m venv clevr

   # For conda
   conda create -n clevr python=3.12
   ```

   Activate the virtual environment:

   ```bash
   # For venv
   source clevr/bin/activate
   # For conda
   conda activate clevr
   ```

2. Install `sherpa` following the top-level Read
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Create Dataset

1. download the Download CLEVR v1.0 (no images) from [the Clevr website](https://cs.stanford.edu/people/jcjohns/clevr/)
2. put the `CLEVR_val_scenes.json` file to the `data` folder
3. Download human created questions from [here](https://cs.stanford.edu/people/jcjohns/iep/)
4. Put the `CLEVR-Humans-val` file to the `data` folder
5. Run `scripts/create_dataset.py` to create the dataset
6. This script will push the dataset to HuggingFace. Update the `--dataset_name` argument when running the experiments to your dataset name

7. The processed dataset is also available on [huggingface](https://huggingface.co/datasets/Dogdays/clevr_subset)

## Setup the Environment Variables
Create a `.env` file in the `clevr-human` folder and copy the content of `.env_template` to it. Then, set the `OPENAI_API_KEY` and `TOGETHER_API_KEY` variables to your OpenAI and TogetherAI API keys, respectively.

## Run Question Answering

Run the `python -m scripts.run_qa ` command to run the question answering task. The command has several arguments to control the behavior of the script. Use the `--help` argument to see the available options:

  * **-h, --help**: show this help message and exit
  * **--dataset_name**: Name of the processed dataset on HuggingFace. Default is `Dogdays/clevr_subset`. You normally don't need to change this unless you have created your own dataset.
  * **--approach**: Scene-based question answering approach to run. One of {direct,state_machine,routing,react}           
   * direct: Directly use the LLM to answer the question
   * state_machine: Use a planning state machine to answer the question
   * routing: Use a routing state machine to answer the question
   * react: Use a ReACT approach to answer the question
  * **--output_folder**: Output file to save the results
  * **--num_processes**: Number of processes to use for parallel processing of the dataset
  * **--log_folder**: Folder to save the execution logs
  * **--llm_type**: Provider of the language model, one of {openai,together}
                        
  * **--llm_name**: Name of the LLM to use. The paper uses the following: gpt-4o (openai), gpt-4o-mini (openai), Qwen/Qwen2.5-7B-Instruct-Turbo (together), Qwen/Qwen2.5-7B-Instruct-Turbo (together) and Meta-Llama-3.1-70B-Instruct-Turbo (together). You can also use other LLM from the two providers. 
  * **--temperature**: Temperature of the LLM, default is 0.01
  * **--num_runs**: Max number of hops when running the state machine
  * **--use_scene**: Whether to use the scene information in the prompt

For example, to run the direct approach with the gpt-4o-mini model, you can run the following command:

```
python -m scripts.run_qa --approach direct --num_processes 1 --llm_type openai --llm_name gpt-4o-mini --log_folder logs/direct/gpt-4o-mini --use_scene --num_runs 10 --output_folder results_new/gpt-4o-mini/run1
```

Or to run the llama-3.1-70B-Instruct-Turbo model with the state machine approach, you can run the following command:

```
python -m scripts.run_qa --approach state_machine --num_processes 1 --llm_type together --llm_name meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --log_folder logs/state_mathine/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --use_scene --num_runs 10 --output_folder results_new/Meta-Llama-3.1-70B-Instruct-Turbo/run1
```

To repeat the experiments in the paper, run each LLM three times with the same command, but change the `--output_folder` argument to save the results in a different folder.

The results will be saved in the `results_new` folder (cached results provided), and the logs will be saved in the `logs` folder. The results will be saved in a JSON file with the name of the LLM and the approach used. Each result file will contain the following fields:
* **predicted**: The predicted answer to the question
* **actual**: The actual answer to the question
* **num_llm_calls**: The number of LLM calls made to answer the question

## Evaluation
The evaluation steps are included in the `evaluation.ipynb` notebook to create tables and figures in the paper for the Clevr-Human use case. To run the notebook, run the following command:

```bash
jupyter notebook evaluation.ipynb
```

Execute all the cells in the notebook to generate the tables and figures. 