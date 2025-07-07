# Sherpa Use Cases
[![DOI](https://zenodo.org/badge/815307724.svg)](https://doi.org/10.5281/zenodo.15824650)

This repo contains artifacts for the paper "SHERPA: A Model-Driven Framework for Large Language Model Execution"

Specifically, it contains the code and data used for the three use cases presented in the paper, as well as a copy of the Sherpa v0.4.0 used in the paper. 

## Artifact Location

This artifact is available at: https://github.com/Aggregate-Intellect/Sherpa4Modeling

The Zenodo DOI for this artifact is: [10.5281/zenodo.15824650](https://doi.org/10.5281/zenodo.15824650)

## Organization


### Description of the approaches
Each approach implemented with Sherpa is modularized as you will see in the use case. Most of the time, each approach contains two components:
* **states.py**: That contains the structure of the state machine
* **actions.py**: That contains the implementation of each actions in the state machine

In some cases, each approach may also contain some helper components, such as:
* **prompts.py**: That contains the prompts used in the approach
* **policy.py**: That contains the customized policy for state selection of the approach

## Installation

### Install Python
To run the code in this repository, you need to install Python. We recommend using a virtual environment such as [venv](https://docs.python.org/3/library/venv.html) or [conda](https://docs.conda.io/en/latest/).

> Unless otherwise specified, this repository has been developed with Python 3.12. Earlier or later versions may also work, but are not guaranteed.

Create the virtual environment:
```bash
# For venv
python -m venv sherpa

# For conda
conda create -n sherpa python=3.12
```

Activate the virtual environment:
```bash
# For venv
source sherpa/bin/activate
# For conda
conda activate sherpa
```

### Install Sherpa
This artifact uses the [Sherpa](https://github.com/Aggregate-Intellect/sherpa) to for the use cases. Specifically, it uses a slightly customized version of Sherpa v0.4.0, which is included in the `sherpa` folder in this repository. You can install Sherpa from the source code in this repository.
To install Sherpa from the source code, first, install with [poetry](https://python-poetry.org/).
```bash
pip install poetry
```

Then, you can run the following commands:
```bash
cd sherpa/src
poetry install --with optional
```

### Install Dependencies
Please refer the `README.md` file in each use case folder for the specific dependencies required for that use case.

### LLMs
This repository uses several APIs for accessing the Large Language Models. You need to set up the API keys for the LLMs you want to use. The supported LLMs are:
- OpenAI: [OpenAI API](https://openai.com/api/)
- TogetherAI: [TogetherAI API](https://www.together.ai/)

## Use cases
> [!NOTE]
> Excepting installing Sherpa, all the instructions for the use cases must be executed in the corresponding use case folder.

the following folder contains material for each use case used in the paper:
* `human_eval` contains the material for the HumanEval benchmark for the code generation use case
* `clevr-human` contains the material for the Clevr-Human dataset for the question answering use case
* `state_based_modeling` contains the material for the class name generation use case

Please refer the `README.md` in each folder for the details of the use case and how to run the experiments.

Each use case contains a `evaluation.ipynb` notebook that contains the steps to use generated results to create tables and figures in the paper.

## Citation
If you found this repository useful, please consider citing the following paper:
```bibtex
@inproceedings{chen2025sherpa,
  author    = {Boqi Chen and
               Kua Chen and
               Jos{\'{e}} Antonio Hern{\'{a}}ndez L{\'{o}}pez and
               Gunter Mussbacher and 
               D{\'{a}}niel Varr{\'{o}} and
               Amir Feizpour},
  title     = {{SHERPA}: A Model-Driven Framework for Large Language Model Execution},
  year      = {2025},
  booktitle = {ACM / IEEE 28th International Conference on Model Driven Engineering Languages and Systems (MODELS),
               2025, Grand Rapids, USA, October 5-10, 2025},
  publisher = {{IEEE}}
}
```

## Acknowledgements
We thank all [contributors](https://github.com/Aggregate-Intellect/sherpa/graphs/contributors) of the SHERPA project for their work on the SHERPA library, which is used in this repository. 

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. The code of the SHERPA project is also licensed under the MIT License, please refer the [license file in the SHERPA project](https://github.com/Aggregate-Intellect/sherpa/blob/main/LICENSE.md) for details.

As the datasets used in this repository are adapted from other projects,  they are subject to their respective licenses. Specifically:
* The [Clevr-Human](https://cs.stanford.edu/people/jcjohns/iep/) dataset is licensed under the Creative Commons CC BY 4.0 license.
* The [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval) dataset is licensed under the MIT License.
* The [State-Based Modeling](https://zenodo.org/records/8118642) is licensed under the Creative Commons CC BY 4.0 license.