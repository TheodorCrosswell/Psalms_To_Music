from pydantic import BaseModel, RootModel
from helpers import (
    get_clean_text,
    get_words,
    get_syllables,
    get_word_options,
    get_syllable_count,
)


class WordSyllable(BaseModel):
    """
    ws = WordSyllable("I've")
    ws.word -> ("i've", "i have")
    ws.syllable -> (1, 2)
    """

    word_options: tuple[str, ...]
    syllable_options: tuple[int, ...]

    def __init__(self, word: str):
        word_options = get_word_options(word)
        syllable_options = tuple(
            get_syllable_count(word_option) for word_option in word_options
        )
        super().__init__(
            word_options=word_options,
            syllable_options=syllable_options,
        )


class SyllablesStrings(RootModel[str]):
    pass


class WordsSyllablesComparison(BaseModel):
    """
    Contains:
    - words_comparison: A list of tuples of words. Each word in
      a tuple is from a different track.
    - syllables_comparison: A list of tuples of syllable options
      for each word. Each tuple of syllable options is from a different track.
    Usage:
    - kjv = WordsSyllables(text=kjv_text_match)
    - query = WordsSyllables(text=hymn_verse)
    - wsc = WordsSyllablesComparison(kjv, query)
    - wsc.words_comparison == [("In", "But"), ("the", "I"), ("beginning", "know")]
    - wsc.syllables_comparison == [((1,), (1,)), ((1,), (1,)), ((3,), (1,)))]
    """

    def __init__(
        self,
        words_syllables_1: WordsSyllables,
        words_syllables_2: WordsSyllables,
    ):
        self.words_comparison = list(
            zip(words_syllables_1.words, words_syllables_2.words)
        )

        self.syllables_comparison = list(
            zip(words_syllables_1.syllables, words_syllables_2.syllables)
        )

    words_comparison: list[tuple[str, str]]
    syllables_comparison: list[tuple[tuple[int, ...], tuple[int, ...]]]
