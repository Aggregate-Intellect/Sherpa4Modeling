We apply for the following artifact evaluation badges:
1. **Available**:
    * The artifact is available on Zenodo with a DOI: https://doi.org/10.5281/zenodo.15824650. This is also specified in the README.md file of the repository.
    * A license file is provided in the repository, as described in the LICENSE.md file.
    * Author information for this artifact is provided in the README.md file of the repository.
2. **Functional**:
    * The artifact can be run with the provided instructions in the README.md file of the repository.
    * All required dependencies for the artifact and uses cases are specified in the `requirements.txt` file for each use case. And tje `Dockerfile` and `docker-compose.yml` files are provided when necessary.
    * All dependencies on external tools (e.g., LLM APIs) are specified in the REQUIREMENTS.md file. It also specifies on other requirements such as Python version and disk space.
    * The instructions to install setting up the environment, installing dependencies, and running the use cases are provided in the README.md file for each use case in this repository, including Docker instructions when necessary.
    * Instructions for running the notebooks to produce the results in the paper are provided in the README.md file for each use case, including the exact table / figure numbers in the paper.
3. **Reusable**:
    * The README.md files contain an organization section that describes the structure of the code and data in the repository.
    * The artifact is organized in a way that allows for easy reuse of the code and data.
        * The code is modularized into the core component `sherpa` as well as the use cases `clevr-human`, `human-eval`, and `state-based-modeling`, so that the code can be reused in other projects for each specific use case.
        * The README.md file in each use case points to the specific files and folders that contain the implementation of the approaches for that use case. Each solution is also organized in a way that allows for easy reuse of the code: states machine definition in `states.py`, actions in `actions.py`, and prompts in `prompts.py` when necessary.
    * While the LLMs used in this artifact depends on specific API providers, in the REQUIREMENTS.md file, we describe how it can be extended to support other LLMs, such as local LLMs using Ollama.
    * Usages for each command that runs the experiments are provided in the README.md file for each use case, including the command line arguments and their descriptions.