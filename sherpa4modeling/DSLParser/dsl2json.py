# %% [markdown]
# # Set up

# %%
import json
from pyparsing import (
    Word, alphas, alphanums, Group, ZeroOrMore, OneOrMore, Optional, delimitedList, Keyword, Suppress, restOfLine, LineEnd, Combine, nums
)

# %% [markdown]
# # Parsing

# %%
from lark import Lark, Transformer
import json

# Define the grammar for the Domain Specific Language (DSL)
dsl_grammar = r"""
    start: enumerations classes relationships

    enumerations: "Enumerations:" enumeration+
    enumeration: "enum" CNAME "(" CNAME ("," CNAME)* ")"

    classes: "Classes:" class_def+
    class_def: CNAME "(" [class_attribute ("," class_attribute)*] ")"
             | CNAME CNAME "(" [class_attribute ("," class_attribute)*] ")"

    class_attribute: CNAME CNAME -> single
                    | CNAME "[]" CNAME -> listing

    relationships: "Relationships:" relationship+
    relationship: mul CNAME CNAME mul CNAME
                | mul CNAME CNAME mul CNAME
                | CNAME CNAME CNAME
    
    mul: NUMBER -> number
        | "*" -> wildcard
        | NUMBER "." ("*"| NUMBER) -> range
    %import common.CNAME
    %import common.NUMBER
    %import common.WS
    %ignore WS
"""

# Create a Lark parser with the defined grammar
parser = Lark(dsl_grammar, start="start")

# Transformer to convert parse tree into a dictionary
class DSLToJSON(Transformer):
    def start(self, items):
        return { "enumerations": items[0], "classes": items[1], "relationships": items[2] }
    
    def enumerations(self, items):
        return { "enumerations": items }
    
    def enumeration(self, items):
        name = items[0]
        values = items[1:]
        return { name: values }
    
    def classes(self, items):
        return {"classes": items}
    
    # def class_def(self, items):
    #     name = items[0]
    #     attributes = items[1:]
    #     if len(attributes) == 1 and isinstance(attributes[0], list):
    #         attributes = attributes[0]
    #     elif len(attributes) == 1 and isinstance(attributes[0], tuple):
    #         attributes = [attributes[0]]
    #     return {name: attributes}
    def class_def(self, items):
        if items[0] == 'abstract':
            name = items[1]
            attributes = items[2:]
            type = "abstract"
            return {name: {"type": "abstract", "attributes": attributes}}
        else:
            type = "regular"
            name = items[0]
            attributes = items[1:]
              
        return {name: {"type": type, "attributes": attributes}}
    
    # def class_attribute(self, items):
    #     # items is a tuple like ('string', 'name')
    #     return {items[1]: items[0]}  # {'name': 'string'}
    
    def single(self, items):
        # items is a tuple like ('string', 'name')
        return {items[1]: items[0]}  # {'name': 'string'}
    
    def listing(self, items):
        # items is a tuple like ('string', 'name')
        return {items[1]: items[0]+"[]"}  # {'name': 'string'}
    
    def relationships(self, items):
        return { "relationships": items }

    def relationship(self, items):
        # print(items)
        if len(items) == 5:
            return {
                "class1_mul": items[0],
                "class 1": items[1],
                "relationship ": items[2],
                "class2_mul": items[3],
                "class 2": items[4]
            }
        elif len(items) == 3:
            return {
                "child": items[0],
                "relationships": items[1],
                "parent": items[2]
            }
    def number(self, items):
        return items[0]

    def wildcard(self, items):
        return "*"
    
    def range(self, items):
        if len(items) == 1:
            return str(items[0]) + ".*"
        elif len(items) == 2:
            return str(items[0]) + "." + str(items[1])

# Sample input DSL
dsl_input = """\
Enumerations:
enum DeviceStatus(Activated, Deactivated)
enum CommandType(lockDoor, turnOnHeating)
enum CommandStatus(Requested, Completed, Failed)
Classes:
SHAS()
SmartHome()
User(string[] name)
Address(string city, string postalCode, string street, string aptNumber)
abstract RuntimeElement(time timestamp)
Relationships:
1 SHAS contain * SmartHome
1 SHAS contain * User
1 SmartHome associate 0..* User
1 SmartHome associate 3..7 User
RelationalTerm inherit BooleanExpression
"""

# Parse the DSL input using the parser
parse_tree = parser.parse(dsl_input)

# Transform the parse tree into JSON
transformer = DSLToJSON()
parsed_dict = transformer.transform(parse_tree)

# Convert the dictionary to a JSON formatted string
dsl_json = json.dumps(parsed_dict, indent=4)

# Print the JSON string
print(dsl_json)



