## General Requirements
In general, the artifact can be run with Python 3.12 or later. However, some use cases may require specific versions of Python or other dependencies. Please refer to the `README.md` file in each use case folder for the specific requirements.

In general, the software requirements are specified in the [README.md](README.md) file in the root directory of this repository. Each use case may have its own specific requirements, which are also specified in the `README.md` file in each use case folder.

The whole repository takes about 30 MB of disk space. 

## LLMs
This repository uses several APIs for accessing the Large Language Models. You need to set up the API keys for the LLMs you want to use. The supported LLMs are:
- OpenAI: [OpenAI API](https://openai.com/api/)
- TogetherAI: [TogetherAI API](https://www.together.ai/)

Note that running LLMs with these APIs will incur costs, so make sure you have enough credits in your account, or using the free tier if available and run a subset of the experiments.

Since the LLM access is implemented using [LangChain](https://python.langchain.com/), you can also use other LLMs supported by LangChain, such as local LLM using [Ollama](https://ollama.com/) with [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama/). In this case, you will need to extend the LLM types manually for each use case: Specifically
* For `clevr-human`, add the new LLM in the `get_llm` method in `clevr_qa/utils.py` 
* For `human_eval`, add the new LLM in the `get_llm` method in `llm_coder/utils.py`
* For `state_based_modeling`, add the new LLM in the `get_llm` method in `modeling/utils.py`

If you use a local LLM, you will need to make sure that your system has enough resources to run the LLM, such as a GPU with VRAM.