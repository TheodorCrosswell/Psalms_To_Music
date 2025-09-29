import polars as pl
from nltk.corpus import cmudict
import re
from pprint import pprint
import pyphen

# Download the cmudict data if you haven't already
# import nltk
# try:
#     nltk.data.find("corpora/cmudict.zip")
# except nltk.downloader.DownloadError:
#     nltk.download("cmudict")

pronunciations = cmudict.dict()
hyphen_gb = pyphen.Pyphen(lang="en_GB")
hyphen_us = pyphen.Pyphen(lang="en_US")


def get_all_kjv_words(kjv_path: str = "../data/kjv.txt") -> list[str]:
    """- Input:
       - kjv_path: the path to kjv.txt, a file with text from all the verses in the KJV Bible concatenated into one text.
    - Output:
       - all_kjv_words: a list of all the unique words found in the KJV Bible. Should be 13725
    """
    with open(kjv_path, "r") as file:
        kjv_text = file.read()

    clean_kjv_text = re.sub(r"[.,:;'()?!|]", "", kjv_text)

    all_kjv_words = set()

    for word in clean_kjv_text.split():
        all_kjv_words.add(word)

    all_kjv_words = list(all_kjv_words)
    return all_kjv_words


def count_syllables_cmudict_shallow(word: str, pronunciations_dict: dict) -> int:
    """Counts the syllables in a word using NLTK's CMUDict.
    If no match is found, it returns -1"""
    word_lower = word.lower()
    if word_lower not in pronunciations_dict:
        return -1
    # A word can have multiple pronunciations; we'll use the first one.
    return len(
        [
            phoneme
            for phoneme in pronunciations_dict[word_lower][0]
            if str.isdigit(phoneme[-1])
        ]
    )


def count_syllables_cmudict_deep(word: str, pronunciations_dict: dict) -> int:
    """Counts the syllables in a word using NLTK's CMUDict.
    If the direct operation fails, it transforms the word in various ways to find a match.
    If there is no match found in CMUDict after transforming the word, it returns -1"""
    # Immediately recognized
    # e.g. "widow" is a direct match in CMUDict
    syllable_count = count_syllables_cmudict_shallow(word, pronunciations_dict)
    if syllable_count > 0:
        return syllable_count

    # Probably a name, as it is not found in the CMUDict and has a capital letter.
    # e.g. "Ashdothpisgah", "A" is not lowercase, and it is a name.
    if not str.islower(word):
        return -1

    # --- Suffix and Transformation Checks ---

    # Rule: Remove 'est', 'eth', 'st', or 'th'
    # e.g. "removeth" -> "remove" + "th"
    # e.g. "staggereth" -> "stagger" + "eth"
    if word.endswith(("eth", "est")):
        base_word_3 = word[:-3]
        syllable_count = count_syllables_cmudict_shallow(
            base_word_3, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count + 1
        base_word_2 = word[:-2]
        syllable_count = count_syllables_cmudict_shallow(
            base_word_2, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count + 1

    # Rule: Remove 's' from end
    # e.g. (not a real example) "preachers" -> "preacher"
    if word.endswith(("s")):
        transformed_word = word[:-1]
        syllable_count = count_syllables_cmudict_shallow(
            transformed_word, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count

    # Rule: Remove 1 extra character from the end (often after suffix removal fails)
    # e.g. "slippeth" -> "slip", which can now be recognized
    if word.endswith(("eth", "est")):
        base_word_4 = word[:-4]
        syllable_count = count_syllables_cmudict_shallow(
            base_word_4, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count + 1

    # Rule: Replace 'iest' or 'ieth' with 'y'
    # e.g. "repliest" -> "reply", which can now be recognized
    # e.g. "blashpemest" -> "blasphemy", which can now be recognized
    if word.endswith(("ieth", "iest", "eth", "est")):
        transformed_word = re.sub(r"(iest|ieth|eth|est)$", "y", word)
        syllable_count = count_syllables_cmudict_shallow(
            transformed_word, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count + 1

    # Rule: Replace British 'our' with 'or'
    # e.g. "honour" -> "honor"
    if "our" in word:
        transformed_word = re.sub(r"our", "or", word)
        syllable_count = count_syllables_cmudict_shallow(
            transformed_word, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count

    # Rule: Replace British 'our' with 'or'
    # e.g. "dishonour" -> "dishonor"
    if "our" in word and word.endswith(("est", "eth")):
        transformed_word = re.sub(r"our\w+", "or", word)
        syllable_count = count_syllables_cmudict_shallow(
            transformed_word, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count + 1

    # Rule: Replace 'or'/'ors' with 'er'
    # e.g. (not a real example) "tormentor" -> "tormenter"
    if word.endswith(("or", "ors")):
        transformed_word = re.sub(r"or\w?", "er", word)
        syllable_count = count_syllables_cmudict_shallow(
            transformed_word, pronunciations_dict
        )
        if syllable_count > 0:
            return syllable_count

    # If no rules matched and the word is unrecognized
    # e.g. "asswage" is not found in CMUDict, even with all the transformations applied.
    return -1


def count_syllables_pyphen(word: str):
    """Uses pyphen to insert hypens into the word, then estimates the number of syllables
    by counting the number of word chunks created.
    Uses both en_US and en_GB, and chooses the maximum value between the two,
    as this seems to be most reliable."""
    clean_word = re.sub("-", "", word)

    hyphenated_word_gb = hyphen_gb.inserted(clean_word)
    syllable_count_gb = len(hyphenated_word_gb.split("-"))

    hyphenated_word_us = hyphen_us.inserted(clean_word)
    syllable_count_us = len(hyphenated_word_us.split("-"))

    return max(syllable_count_us, syllable_count_gb)


def get_syllable_count(word: str):
    """Gets the syllable count from CMUDict with a deep comparison.
    As a fallback, uses pyphen to estimate the number of syllables."""
    syllable_count = count_syllables_cmudict_deep(word, pronunciations)
    if syllable_count == -1:
        syllable_count = count_syllables_pyphen(word)
    return syllable_count
