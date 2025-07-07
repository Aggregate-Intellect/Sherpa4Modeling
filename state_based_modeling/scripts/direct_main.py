import os
from argparse import ArgumentParser
from multiprocessing import Pool

import dotenv
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger
from modeling.utils import get_llm
from tqdm import tqdm

dotenv.load_dotenv()

prompt = """
## Problem desription
{description}

## Instruction
Identify classes and enumerations given a problem description.
A class is the description for a set of similar objects that have the same structure and behavior, i.e., its instances
All objects with the same features and behavior are instances of one class.
In general, something should be a class if it could have instances.
In general, something should be an instance if it is clearly a single member of the set defined by a class.
Keep in mind that some of the nouns may be attributes or roles of the identified classes.
Choose proper names for classes according the the following rules:
1. Noun
2. Singular
3. Not too general, not too specific – at the right level of abstraction
4. Avoid software engineering terms (data, record, table, information)
5. Conventions: first letter capitalized; camel case without spaces if needed

Example class names:
Hospital, Doctor, PartTimeEmployee

Constraints:
Create classes at the right level of abstraction.
Not all nouns in the nouns list are classes, some of them may be attributes, role names, or even not needed for diagram.
Do NOT include all the nouns list as classes. Evaluate if it is needed to be a class.
ONLY generate classes that are necessary to develop the system.


Example:
Problem Description: This system helps the Java Valley police officers keep track of the cases they are assigned to do. Officers may be assigned to investigate particular crimes, which involves interviewing victims at their homes and entering notes in the PI system. Crime can either be solved, pending or unsolved.
Result:
PISystem()
PoliceStation()
Case()
PoliceOfficer() 
Victim()
Crime()
Note()
enum CrimeStatus(Solved, Pending, Unsolved)

Only output class names and enumeration values followed by \"()\", for enumerations, also add "enum" before the name and add the enumeration literal in \"()\".
Separate classes and enumerations by newlines and do not include any other words or symbols in your generated text.
"""  # noqa E501


def generate_for_problem(inputs):
    description, name, output_folder, args = inputs

    llm = get_llm(args.model_type, args.llm)
    prompt_template = ChatPromptTemplate([("user", prompt)])
    chain = prompt_template | llm
    result = chain.invoke(
        {
            "description": description,
        }
    ).content

    output_file_path = os.path.join(output_folder, name + ".txt")
    with open(output_file_path, "w") as file:
        file.write(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--llm", type=str, help="Model name", required=True)
    parser.add_argument("--model_type", type=str, help="Model type", required=True)
    parser.add_argument(
        "--output_folder", type=str, help="Output folder", default="results"
    )
    parser.add_argument(
        "--run_number", type=int, help="Run number", default=1, required=True
    )
    parser.add_argument(
        "--num_processes", type=int, help="Number of processes", default=1
    )

    args = parser.parse_args()

    model_name = args.llm
    if "/" in model_name:
        subfolder = (
            f"{args.output_folder}/"
            + model_name.split("/")[1]
            + f"_output_direct/run{args.run_number}"
        )
    else:
        subfolder = (
            f"{args.output_folder}/"
            + model_name
            + f"_output_direct/run{args.run_number}"
        )

    os.makedirs(subfolder, exist_ok=True)

    data = []
    df = pd.read_csv("modeling_problems.csv")
    for index, row in df.iterrows():
        description = row["description"]
        name = str(index) + "_" + row["name"].strip()
        data.append((description, name, subfolder, args))

    logger.info(f"Starting generation for {len(data)} problems with {args.llm}")
    with Pool(args.num_processes) as p:
        # NOTE: the order is not important in this case since the output is written to
        # a file with the name
        list(tqdm(p.imap_unordered(generate_for_problem, data), total=len(data)))

    logger.info("Finished!")
