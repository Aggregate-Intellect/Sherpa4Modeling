# Running Sherpa for Generating Model Classes

To run Sherpa for generating model classes, use the following command:

```bash
python main_class.py
```

You need to set OPENAI_API_KEY in the terminal by:
```
export OPENAI_API_KEY=your_api_key_here (Mac)
set OPENAI_API_KEY=your_api_key_here (Windows)
```

To generate a complete model using the Yujing Approach, run:
```
python main_yujing_modified.py
```


The file modeling_problems.csv contains descriptions and titles for 8 modeling problems. Use this file as input for generating models.