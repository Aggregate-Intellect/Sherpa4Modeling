from pydantic import BaseModel
import re


class DomainModel(BaseModel):
    classes: list[str]
    enums: list[str]

    def __str__(self):
        classes_str = "\n".join(self.classes)
        enums_str = "\n".join(self.enums)

        return f"Enumerations:\n{enums_str}\n\nClasses:\n{classes_str}"


class ClassNameOutputTransform:
    pattern: str = r"(abstract\s+|enum\s+)?([A-Za-z]\w*)\s*\((.*?)\)"

    def transform_output(self, output: str) -> DomainModel:
        matches = re.findall(self.pattern, output)
        classes = []
        enums = []

        for match in matches:
            if match[0].strip() == "enum":
                enums.append(f"{match[1]}({match[2]})")
            else:
                classes.append(f"{match[1]}()")

        return DomainModel(classes=classes, enums=enums)


