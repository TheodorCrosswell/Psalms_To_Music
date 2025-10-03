from models import WordsSyllables


def read_text_into_WordsSyllables(path: str = "../data/kjv.txt") -> str:
    with open(path, "r") as file:
        text = str(file.read())
    words_syllables = WordsSyllables(text=text)
    return words_syllables
