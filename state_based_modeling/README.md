# Sherpa for Modeling (The Class Name Generation Use Case)
Designing state machines for solving the class name generation task in the Modeling dataset

## Organization

## Installation
1. Create a new virtual environment for this use case (not required, but recommended)

   ```bash
   # For venv
   python -m venv modeling

   # For conda
   conda create -n modeling python=3.10
   ```

   Activate the virtual environment:

   ```bash
   # For venv
   source modeling/bin/activate
   # For conda
   conda activate modeling
   ```

2. Install `sherpa` following the top-level Read
3. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Setup the Environment Variables
Create a `.env` file in the `clevr-human` folder and copy the content of `.env_template` to it. Then, set the `OPENAI_API_KEY` and `TOGETHER_API_KEY` variables to your OpenAI and TogetherAI API keys, respectively.

## Run Class Name generation
### Run the Direct Approach
Run `scripts/direct_main.py` to generate class names using LLMs only. It contains the following commands

* **-h, --help**: show this help message and exit
* **--model_type**: Provider of the model to use. One of {openai, together}
* **--llm**: Name of the LLM to use. The paper uses the following: gpt-4o (openai), gpt-4o-mini (openai), Qwen/Qwen2.5-7B-Instruct-Turbo (together), Qwen/Qwen2.5-7B-Instruct-Turbo (together) and Meta-Llama-3.
* **--run_number**: Run number to use for the experiment. This is used to create a unique output folder for the results.
* **--output_folder**: Output folder to save the results, default is `results_new`
* **--num_processes**: Number of processes to use for parallel processing of the dataset, default is 8

For example, to run the direct approach with the gpt-4o-mini model, you can run the following command:

```bash
python -m scripts.direct_main --model_type=openai --llm=gpt-4o-mini --run_number 1 --num_processes 1
```

### Run the State Machine Approaches
Run `scripts.sm_main.py` to generate class names using the state machine approaches. It contains the following commands:
* **-h, --help**: show this help message and exit
* **--model_type**: Provider of the model to use. One of {openai, together}
* **--llm**: Name of the LLM to use. The paper uses the following: gpt-4o (openai), gpt-4o-mini (openai), Qwen/Qwen2.5-7B-Instruct-Turbo (together), Qwen/Qwen2.5-7B-Instruct-Turbo (together) and Meta-Llama-3.
* **--run_number**: Run number to use for the experiment. This is used to create a unique output folder for the results.
* **--output_folder**: Output folder to save the results, default is `results_new`
* **--num_processes**: Number of processes to use for parallel processing of the dataset, default is 8
* **--approach**: Approach to use for generating class names. One of {sherpa, sherpa_mig}
    * Sherpa runs the Inspection State machine in the paper
    * sherpa_mig runs the MIG state machine in the paper


For example, to run the Inspection state machine with the gpt-4o-mini model, you can run the following command:
```bash
python scripts.sm_main --model_type=openai --llm=gpt-4o-mini --run_number 1 --approach sherpa
```

## Evaluation
The evaluation requires Python 3.8. First install Python 3.8 with a virtual environment. Then install the requirements:
```bash
pip install -r requirements.txt
```

Then run the `evaluation.ipynb` notebook. It will evaluate the generated classes and print the results. 