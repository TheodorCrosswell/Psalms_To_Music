from nltk.corpus import cmudict
import re
import pyphen
from rapidfuzz import fuzz, process

# Download the cmudict data if you haven't already
# import nltk
# try:
#     nltk.data.find("corpora/cmudict.zip")
# except nltk.downloader.DownloadError:
#     nltk.download("cmudict")

pronunciations = cmudict.dict()
hyphen_gb = pyphen.Pyphen(lang="en_GB")
hyphen_us = pyphen.Pyphen(lang="en_US")


# TODO: synchronize this and the cleaning that occours in WordSyllable (disorganized)
def get_clean_text(text: str) -> str:
    """
    Lowercases the text, removes the punctuation marks, but preserves apostrophes
     in the middle of words and at the end of words that end with 's'.
    e.g. :
    - Homie's -> homie's
    - Homies' " -> homies's
    - 'Homies' -> homies
    """
    apostrophes = f"{chr(39)}{chr(8217)}{chr(700)}{chr(8216)}{chr(8219)}"
    lowercase = text.lower()
    reformat_apostrophes = re.sub(f"[{apostrophes}]", "'", lowercase)
    remove_puncutation_except_apostrophes = re.sub(
        r"[^\w\s^']", "", reformat_apostrophes
    )
    remove_bad_apostrophes = re.sub(
        r"'(\w+)'", r"\g<1>", remove_puncutation_except_apostrophes
    )
    add_s_to_possesive_apostrophes = re.sub(
        r"(\w+')(\s)", r"\g<1>s\g<2>", remove_bad_apostrophes
    )
    clean_text = add_s_to_possesive_apostrophes
    return clean_text


def get_words(clean_text: str) -> list[str]:
    words = clean_text.split()
    return words


# TODO: convert it to return tuples of all syllable pronunciation options.
# e.g. "I've" -> (1,2), because it can be pronounced as either "I've" or as "I have"
def count_syllables_cmudict_shallow(word: str, pronunciations_dict: dict) -> int:
    """
    Counts the syllables in a word using NLTK's CMUDict.
    If no match is found, it returns -1
    """
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
    """
    Counts the syllables in a word using NLTK's CMUDict.
    If the direct operation fails, it transforms the word in various ways to find a match.
    If there is no match found in CMUDict after transforming the word, it returns -1
    """
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


def count_syllables_pyphen(word: str) -> int:
    """
    Uses Pyphen to insert hypens into the word, then estimates the number of syllables
    by counting the number of word chunks created.
    Uses both en_US and en_GB, and chooses the maximum value between the two,
    as this seems to be most reliable.
    """
    clean_word = re.sub("-", "", word)

    hyphenated_word_gb = hyphen_gb.inserted(clean_word)
    syllable_count_gb = len(hyphenated_word_gb.split("-"))

    hyphenated_word_us = hyphen_us.inserted(clean_word)
    syllable_count_us = len(hyphenated_word_us.split("-"))

    return max(syllable_count_us, syllable_count_gb)


def get_syllable_count(input_word: str) -> int:
    """
    Gets the syllable count from CMUDict with a deep comparison.
    As a fallback, uses pyphen to estimate the number of syllables.
    """
    input_word = input_word.lower()
    # If the word has a space, count the syllables for each word and add it up.
    # e.g. "I have" -> 1 + 1 -> 2
    words = input_word.split()
    total_syllable_count = 0
    for word in words:
        syllable_count = count_syllables_cmudict_deep(word, pronunciations)
        if syllable_count == -1:
            syllable_count = count_syllables_pyphen(word)
        total_syllable_count += syllable_count
    return total_syllable_count


# TODO: implement this in a function with syllable counting, to avoid hyphenating the same word twice
def hyphenate_word(
    word: str, syllable_count: int | str, hyphenator: pyphen.Pyphen = hyphen_gb
) -> str:
    """
    Hyphenates a word based on a given syllable count.
    It first tries to use the Pyphen library. If the result does not match the
    syllable_count, it falls back to a method that divides the word into
    roughly equal parts.
    """
    # Arbitrary decision: use Pyphen(lang="en_GB") as opposed to "en_US"
    # Let Pyphen attempt to hyphenate the word first.
    hyphenated_word = hyphenator.inserted(word)

    syllable_count = int(syllable_count)

    # Check if Pyphen's hyphenation matches the desired syllable count.
    # The number of hyphens should be one less than the number of syllables.
    if hyphenated_word.count("-") == syllable_count - 1:
        return hyphenated_word
    else:
        # If Pyphen's output is not correct, use a fallback method.
        # For a single syllable word, no hyphens are needed.
        if syllable_count <= 1:
            return word

        word_len = len(word)
        step = word_len / syllable_count

        parts = []
        last_cut = 0
        for i in range(1, syllable_count):
            # Calculate the position for the next hyphen.
            cut = round(i * step)
            parts.append(word[last_cut:cut])
            last_cut = cut
        parts.append(word[last_cut:])
        full_word = "-".join(parts)
        return full_word


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

# TODO: deprecate. Not sure that this will ever get used vs. just using WordsSyllables("string")
# def get_word_options(word: str) -> tuple[str, ...]:
#     """Expands contractions and common slang in a given word.

#     This function takes a word and returns a tuple of possible
#     expansions. It handles common English contractions and some informal
#     slang words.

#     Input:
#         - clean_word: A single word, without spaces.
#             - e.g. "y'all'd"

#     Output:
#         - A tuple[str, ...] containing the original word and its possible
#         expansions.
#             - e.g. "you all would"
#     """
#     clean_word = word.lower()
#     word_options = set((clean_word,))
#     word_options.update(
#         get_word_options_specific_cases.get(
#             clean_word,
#             get_word_options_specific_cases.get(
#                 clean_word.replace("'", ""),
#                 (),
#             ),
#         )
#     )

#     for ending, changed_ending in get_word_options_general_rules.items():
#         word_options_tuple = tuple(word_options)
#         for word_option in word_options_tuple:
#             word_options.add(word_option.replace(ending, changed_ending).strip())
#     return tuple(word_options)


def get_fuzzy_matches(
    search_syllables_str: str, kjv_syllables_str: str, score_cutoff: float
) -> list[tuple[str, float, int]]:
    """
    This function scans the entire kjv_syllables_str for matches with search_syllables_str.
    Returns matches that have a score above the score_cutoff.

    - Input:
        - search_syllables_str: The short string to be a query
            - e.g. '1111112113111211111321211'
        - kjv_syllables_str: The long string to be searched
            - e.g. '1132111111' + (...) + '111112111112'
        - score_cutoff: The minimum score that will be returned
            - e.g. 95.0
    - Output
        - list[tuple[
            - matching_section: The string matching the query string
                - e.g. '1111112113111211111132121'
            - score: The score of this match
                - e.g. 96.0
            - index: The index of the first word in the kjv_syllables_str that matched the search_syllables_str
                - e.g. 757980
            ]]
    """
    step = 1
    search_length = len(search_syllables_str)
    full_length = len(kjv_syllables_str)
    result = process.extract(
        query=search_syllables_str,
        choices=(
            kjv_syllables_str[i : i + search_length]
            for i in range(0, full_length - search_length + 1, step)
        ),
        scorer=fuzz.ratio,
        limit=None,
        score_cutoff=score_cutoff,
    )
    return result
