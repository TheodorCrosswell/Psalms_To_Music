from pydantic import BaseModel, RootModel
from helpers import (
    get_word_options_specific_cases,
    get_word_options_general_rules,
    get_syllable_count,
    hyphenate_word,
)


class WordSyllable(BaseModel):
    """
    A representation of a word, with 2 attributes.
        - word_options: the hyphenated variants of the word given.
        - syllable_options: the syllable counts of the respective word_options.
    Example:
        sentence = "Cameron's also known as 'Big C'."
        words = sentence.split()
        word1 = words[0] # "Cameron's"
        ws = WordSyllable(word1)
        ws.word_options -> ("cameron's","cameron is")
        ws.syllable_options -> (3, 4)
    """

    word_options: tuple[str, ...]
    syllable_options: tuple[int, ...]

    def __init__(
        self,
        word: str | None = None,
        word_options: tuple[str, ...] | None = None,
        syllable_options: tuple[int, ...] | None = None,
    ):
        """Expands contractions and common slang in a given word.

        This function takes a word and returns a tuple of possible
        expansions. It handles common English contractions and some informal
        slang words.

        Input:
            - word: A word or several words separated by " ".
                - e.g. "Cameron's"

        Output:
            - A WordSyllable containing the original word and its possible
            expansions.
                - e.g. WordSyllable(word_options=('ca-mer-on is', "cam-ero-n's"), syllable_options=(4, 3))
        """
        if word and not word_options and not syllable_options:
            clean_word = word.lower()
            word_options = set((clean_word,))
            word_options.update(
                get_word_options_specific_cases.get(
                    clean_word,
                    get_word_options_specific_cases.get(
                        clean_word.replace("'", ""),
                        (),
                    ),
                )
            )

            for ending, changed_ending in get_word_options_general_rules.items():
                word_options_tuple = tuple(word_options)
                for word_option in word_options_tuple:
                    word_options.add(
                        word_option.replace(ending, changed_ending).strip()
                    )

            # TODO: optimization: speedup by storing results for words already processed.
            hyphenated_word_options = []
            word_option_syllable_counts = []
            for word_option in word_options:
                individual_words = word_option.split()
                hyphenated_words = []
                word_option_syllable_count = 0
                for word in individual_words:
                    word_syllable_count = get_syllable_count(word)
                    word_option_syllable_count += word_syllable_count
                    word_hyphenated = hyphenate_word(word, word_syllable_count)
                    hyphenated_words.append(word_hyphenated)
                hyphenated_word_option = " ".join(hyphenated_words)
                hyphenated_word_options.append(hyphenated_word_option)
                word_option_syllable_counts.append(word_option_syllable_count)
            super().__init__(
                word_options=hyphenated_word_options,
                syllable_options=word_option_syllable_counts,
            )
        if not word and word_options and syllable_options:
            super().__init__(
                word_options=word_options,
                syllable_options=syllable_options,
            )
