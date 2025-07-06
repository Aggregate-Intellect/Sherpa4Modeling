# Running Sherpa for Generating Model Classes

Make sure to first install Sherpa using the instruction from the top-level repository.
To run Sherpa for generating model classes, use the following command:

```bash
python main_class.py --model_type=openai --llm=gpt-4o-mini --run_number 1 --approach sherpa
```

- `main_class.py` is a driver. It serves as the main entry point for the application.
- `model_class.py` defines the actual state machine. It contains the logic for managing the states and transitions.
- `action.py` contains actions used in the state machine. These actions are executed based on the current state.

You need to set OPENAI_API_KEY in the terminal by:
```bash
export OPENAI_API_KEY=your_api_key_here (Mac)
set OPENAI_API_KEY=your_api_key_here (Windows)
```

To generate class names with the MIG state machine, using the following command
```bash
python main_class.py --model_type=openai --llm=gpt-4o-mini --run_number 1 --approach sherpa_mig
```


The following comment runs for a direct generation approach:
```bash
python direct_prompt.py --model_type=openai --llm=gpt-4o --run_number 3
```

## Evaluation
The evaluation requires Python 3.8. First install Python 3.8 with a virtual environment. Then install the requirements:
```bash
pip install -r requirements.txt
```

Then run the `evaluation.ipynb` notebook. It will evaluate the generated classes and print the results. 