from __future__ import annotations
from pydantic import BaseModel, RootModel
from helpers import (
    get_syllable_count,
    hyphenate_word,
    get_fuzzy_matches,
    get_syllables,
    get_clean_text,
    get_words,
)


get_word_options_general_rules = {
    "'d": " would",
    "'ll": " will",
    "'re": " are",
    "'ve": " have",
    "n't": " not",
    "'s": " is",
    "'m": " am",
    "y'all": " you all",
}
get_word_options_specific_cases = {
    "boutta": ("about to",),
    "aboutta": ("about to",),
    "ain't": ("am not", "is not", "are not", "has not", "have not"),
    "can't": ("cannot",),
    "he'd": ("he would", "he had"),
    "he's": ("he is", "he has"),
    "i'd": ("i would", "i had"),
    "it's": ("it is", "it has"),
    "let's": ("let us",),
    "shan't": ("shall not",),
    "she'd": ("she would", "she had"),
    "she's": ("she is", "she has"),
    "that's": ("that is", "that has"),
    "there's": ("there is", "there has"),
    "they'd": ("they would", "they had"),
    "we'd": ("we would", "we had"),
    "what's": ("what is", "what has"),
    "where's": ("where is", "where has"),
    "who'd": ("who would", "who had"),
    "who's": ("who is", "who has"),
    "won't": ("will not",),
    "you'd": ("you would", "you had"),
    "gimme": ("give me",),
    "gonna": ("going to",),
    "wanna": ("want to",),
    "gotta": ("got to",),
    "hafta": ("have to",),
    "dunno": ("don't know",),
    "lemme": ("let me",),
    "kinda": ("kind of",),
    "sorta": ("sort of",),
    "outta": ("out of",),
    "c'mon": ("come on",),
    "shoulda": ("should have",),
    "coulda": ("could have",),
    "woulda": ("would have",),
    "musta": ("must have",),
    "mighta": ("might have",),
    "shouldna": ("should not have",),
    "couldna": ("could not have",),
    "wouldna": ("would not have",),
    "whatcha": ("what are you", "what have you"),
    "betcha": ("bet you",),
    "gotcha": ("got you",),
    "dontcha": ("don't you",),
    "didntcha": ("didn't you",),
    "wontcha": ("won't you",),
    "need'a": ("need to",),
    "oughta": ("ought to",),
    "supposta": ("supposed to",),
    "useta": ("used to",),
    "lotta": ("lot of",),
    "cuppa": ("cup of",),
    "s'more": ("some more",),
    "tellem": ("tell them",),
    "i'mma": ("i'm going to",),
    "y'all": ("you all",),
    "y'all'd've": ("you all would have",),
    "amn't": ("am not",),
    "'tis": ("it is",),
    "'twas": ("it was",),
    "o'er": ("over",),
    "ne'er": ("never",),
    "e'er": ("ever",),
    "e'en": ("even",),
}


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


class WordsSyllables(BaseModel):
    words: list[str]
    syllables: str

    def __init__(self, text: str):
        clean_text = get_clean_text(text)
        words = get_words(clean_text)
        syllables = get_syllables(words)
        super().__init__(
            words=words,
            syllables=syllables,
        )

    def find_matches(self, words_syllables: WordsSyllables, score_cutoff: float):
        """Make sure to call this from the shorter WordsSyllables, passing the longer WordsSyllables as a parameter"""
        matches = get_fuzzy_matches(
            self.syllables, words_syllables.syllables, score_cutoff
        )
        results = []
        for match in matches:
            start_index = match[2]
            end_index = start_index + len(match[0])
            results.append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "similarity": match[1],
                    "words_matched": words_syllables.words[start_index:end_index],
                }
            )
        return results
