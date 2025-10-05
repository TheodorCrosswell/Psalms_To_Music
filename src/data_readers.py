from models import WordsSyllables


def read_text(path: str = "../data/kjv.txt") -> str:
    with open(path, "r") as file:
        text = str(file.read())
    return text
