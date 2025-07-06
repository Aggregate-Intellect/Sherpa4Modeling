from typing import Optional

from loguru import logger
from modeling.output_transformer import ClassNameOutputTransform


class DomainModel:

    def __init__(
        self,
        filename: Optional[str] = None,
        model_str: Optional[str] = None,
        transform: bool = False,
    ):
        # filename and model_str cannot be both None
        if filename is None and model_str is None:
            raise ValueError("Both filename and model_str cannot be None")

        if filename is not None:
            with open(filename, "r") as f:
                model_str = f.read()

        self.model_str = model_str.split("Relationships:")[0]
        if transform:
            self.model_str = str(
                ClassNameOutputTransform().transform_output(self.model_str)
            )

        if len(model_str.split("Relationships:")) > 1:
            self.edges_str = model_str.split("Relationships:")[1]
        else:
            self.edges_str = ""
        self.set_classes()
        self.set_attributes()
        self.set_edges()

        self.cls_names.remove("Enumerations:")
        self.cls_names.remove("Classes:")
        self.cls_raw.remove("Enumerations:")
        self.cls_raw.remove("Classes:")

        self.rel = None

    def set_classes(self):
        self.cls_names = []
        tmp = self.model_str.strip().splitlines()
        self.cls_names = [i.split("(")[0].strip() for i in tmp if len(i) > 0]

    def set_attributes(self):
        tmp = self.model_str.strip().splitlines()
        self.cls_raw = [i.strip() for i in tmp if len(i) > 0]

        self.cls_attrs = {}
        print(self.cls_raw)
        regular_index = self.cls_raw.index("Classes:")

        for cla, dsl in zip(self.cls_names, self.cls_raw):
            if dsl == "Enumerations:" or dsl == "Classes:":
                continue
            else:
                index = self.cls_raw.index(dsl)
                if index > regular_index:
                    class_type = "regular"
                    if "abstract" in dsl:
                        class_type = "abstract"
                else:
                    class_type = "enum"
                self.cls_attrs[cla] = {
                    "score": 0,
                    "type": class_type,
                    "dsl": dsl,
                    "counterpart": None,
                    "attributes": {},
                }

                attributes = dsl.split("(")[1][:-1].split(",")
                for attr in attributes:
                    attr = attr.strip()
                    if len(attr) > 0:
                        self.cls_attrs[cla]["attributes"][attr] = {
                            "score": 0,
                            "counterpart": None,
                        }

    def set_edges(self):
        self.edges_str = self.edges_str.replace("0..*", "*")
        tmp = self.edges_str.strip().splitlines()
        self.rel_raw = [i.strip() for i in tmp if len(i) > 0]

        for i in self.rel_raw:
            length = len(i.split())
            if length == 5 or length == 3:
                pass
            else:
                logger.debug("Reference Length error:", i)

        self.relations = []
        for e in self.rel_raw:
            self.relations.append({"dsl": e, "score": 0, "counterpart": None})

    def find_relation(self, dsl):
        for index, i in enumerate(self.relations):
            if i["dsl"] == dsl and i["counterpart"] is None:
                return index, i

        return None
