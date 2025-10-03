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


def get_syllables(words: list[str]) -> list[tuple[int, ...]]:
    syllables = []
    for word in words:
        syllable_counts = get_syllable_counts(word)
        syllables.append(syllable_counts)
    return syllables


# # Don't use this. Use WordsSyllables(text="example text") instead.
# def get_WordsSyllables(text: str) -> WordsSyllables:
#     return WordsSyllables(text=text)


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
):
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
        return "-".join(parts)


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


def get_word_options(word: str) -> tuple[str, ...]:
    """Expands contractions and common slang in a given word.

    This function takes a word and returns a tuple of possible
    expansions. It handles common English contractions and some informal
    slang words.

    Input:
        - clean_word: A single word, without spaces.
            - e.g. "y'all'd"

    Output:
        - A tuple[str, ...] containing the original word and its possible
        expansions.
            - e.g. "you all would"
    """
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
            word_options.add(word_option.replace(ending, changed_ending).strip())
    return tuple(word_options)


# TODO: make this get the syllable count at the same time and hyphenate the words
def get_WordSyllable_options(word: str) -> tuple[str, ...]:
    """Expands contractions and common slang in a given word.

    This function takes a word and returns a tuple of possible
    expansions. It handles common English contractions and some informal
    slang words.

    Input:
        - clean_word: A single word, without spaces.
            - e.g. "y'all'd"

    Output:
        - A tuple[str, ...] containing the original word and its possible
        expansions.
            - e.g. "you all would"
    """
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
            word_options.add(word_option.replace(ending, changed_ending).strip())
    return tuple(word_options)


# TODO: deprecate this. I do not need just the syllables, so this is not useful.
def text_to_syllables_str(text: str) -> str:
    """
    Converts a text to a syllable count string
    - e.g. "In the beginning God created the heaven and the earth" -> '1132111111'
    - In: 1 the: 1 beginning: 3 God: 1 created: 3 the: 1 heaven: 2 and: 1 the: 1 earth: 1
    """
    clean_text = get_clean_text(text)
    words = clean_text.split()
    syllables_str = "".join([str(get_syllable_count(word)) for word in words])
    return syllables_str


# TODO: deprecate this. Use a WordsSyllables class instead.
#  e.g. WordsSyllables.__init__(text: str)
def text_to_words_syllables_str(text: str) -> tuple[list[str], str]:
    """
    Converts a text to a syllable count string
    - e.g. "In the beginning God created the heaven and the earth" -> '1132111111'
    - In: 1 the: 1 beginning: 3 God: 1 created: 3 the: 1 heaven: 2 and: 1 the: 1 earth: 1
    """
    clean_text = get_clean_text(text)
    words = clean_text.split()
    syllables_str = "".join([str(get_syllable_count(word)) for word in words])
    return words, syllables_str


# TODO: deprecate this. This is probably useless
def get_exact_matches(main_syllables: str, search_syllables: str) -> list[int]:
    """
    Gets the locations of all exact matches by iteratively
    comparing each element of search_syllables to the elements
    in main_syllables, skipping the rest of the check if a mismatch is found.

    Accepts input as:
    - "1132111111" (str)
    """
    match_locations = []
    for i in range(0, len(main_syllables)):
        match_flag = True
        for j in range(0, len(search_syllables)):
            # Not a match, break out of the loop and skip checking the rest for this position.
            if main_syllables[i + j] != search_syllables[j]:
                match_flag = False
                break
        if match_flag:
            match_locations.append(i)
    return match_locations


# TODO: deprecate this. This is probably useless
def get_exact_matches_optimized(
    main_syllables: str, search_syllables: str
) -> list[int]:
    """
    Gets the locations of all exact matches by iteratively
    comparing each element of search_syllables to the elements
    in main_syllables, skipping the rest of the check if a mismatch is found.

    Optimization: Finds the max value of search_syllables, then searches for that max value,
    and facilitates a faster search, because the max value is rarer than 1s or 2s

    In testing, this optimization resulted in 100% speedup vs non-optimized version: 0.16s -> 0.08s

    Accepts input as:
    - "1132111111" (str)
    """
    match_locations = []
    peak = max(search_syllables)
    peak_index = search_syllables.index(peak)
    for i in range(0, len(main_syllables)):
        if i + peak_index > len(main_syllables) - 1:
            break
        elif main_syllables[i + peak_index] != peak:
            continue
        match_flag = True
        for j in range(0, len(search_syllables)):
            # Not a match, break out of the loop and skip checking the rest for this position.
            if main_syllables[i + j] != search_syllables[j]:
                match_flag = False
                break
        if match_flag:
            match_locations.append(i)
    return match_locations


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


# TODO: deprecate. Outputting the results to text is a bad idea. It should return an object instead.
def match_results_to_text(
    words_long: list[str],
    words_short: list[str],
    syllables_long: str,
    syllables_short: str,
    index: int = 0,
    hyphenator: pyphen.Pyphen = hyphen_gb,
) -> str:
    """
    - Input:
        - words_long: long list of words
            - e.g. ['In', 'the', 'beginning', 'God', 'created'], (...), ['be', 'with', 'you', 'all', 'Amen']
        - words_short: short list of words
            - e.g. ['But', 'I', 'know', 'whom', 'I', 'have', 'believed', 'and', 'am', 'persuaded']
        - syllables_long: string, each digit is the number of syllables in the correspoding word
            - e.g. "11313" + (...) + "11112"
        - syllables_short: string, each digit is the number of syllables in the correspoding word
            - e.g. "1111112113"
        - index: int, the index of the match in words_long/syllables_long
            - e.g. 757980
        - hyphenator: pyphen.Pyphen = hyphen_gb,
            - e.g. pyphen.Pyphen(lang="en_GB")
    - Output:
        - output_string: string
            - e.g. (you have to open this file to see it properly)

                   I                Ive       / 1   1
                 have       X   com-mit-ted   / 1 X 3
              com-mit-ted   X      un-to      / 3 X 2
                 un-to      X       him       / 2 X 1
                  him       X    agai-nst     / 1 X 2
               agai-nst     X      that       / 2 X 1
                  that              day       / 1   1

    """
    validate_words_syllables_search_parameters(
        words_long, words_short, syllables_long, syllables_short
    )
    output_string = ""
    for i in range(len(words_short)):
        word_long = words_long[index + i]
        word_short = words_short[i]
        syllable_long = syllables_long[index + i]
        syllable_short = syllables_short[i]
        if syllable_long != "1":
            word_long = hyphenate_word(word_long, syllable_long, hyphenator)
        if syllable_short != "1":
            word_short = hyphenate_word(word_short, syllable_short, hyphenator)
        indicator = " "
        if syllable_long != syllable_short:
            indicator = "X"
        output_string += f"{word_long:^15} {indicator} {word_short:^15} / {syllable_long} {indicator} {syllable_short}\n"
    return output_string


# # TODO: make this return a pythonic object, not JSON.
# def match_results_to_json(
#     words_long: list[str],
#     words_short: list[str],
#     syllables_long: str,
#     syllables_short: str,
#     index: int = 0,
#     hyphenator: pyphen.Pyphen = hyphen_gb,
# ) -> list[WordDetail]:
#     """
#     - Input:
#         - words_long: long list of words
#             - e.g. ['In', 'the', 'beginning', 'God', 'created'], (...), ['be', 'with', 'you', 'all', 'Amen']
#         - words_short: short list of words
#             - e.g. ['But', 'I', 'know', 'whom', 'I', 'have', 'believed', 'and', 'am', 'persuaded']
#         - syllables_long: string, each digit is the number of syllables in the correspoding word
#             - e.g. "11313" + (...) + "11112"
#         - syllables_short: string, each digit is the number of syllables in the correspoding word
#             - e.g. "1111112113"
#         - index: int, the index of the match in words_long/syllables_long
#             - e.g. 757980
#         - hyphenator: pyphen.Pyphen = hyphen_gb,
#             - e.g. pyphen.Pyphen(lang="en_GB")
#     - Output:
#         - output_string: string
#             - e.g. (you have to open this file to see it properly)

#                    I                Ive       / 1   1
#                  have       X   com-mit-ted   / 1 X 3
#               com-mit-ted   X      un-to      / 3 X 2
#                  un-to      X       him       / 2 X 1
#                   him       X    agai-nst     / 1 X 2
#                agai-nst     X      that       / 2 X 1
#                   that              day       / 1   1

#     """
#     validate_words_syllables_search_parameters(
#         words_long, words_short, syllables_long, syllables_short
#     )
#     output = []
#     for i in range(len(words_short)):
#         word_long = words_long[index + i]
#         word_short = words_short[i]
#         syllable_long = syllables_long[index + i]
#         syllable_short = syllables_short[i]
#         if syllable_long != "1":
#             word_long = hyphenate_word(word_long, syllable_long, hyphenator)
#         if syllable_short != "1":
#             word_short = hyphenate_word(word_short, syllable_short, hyphenator)
#         output.append(
#             {
#                 "word_long": word_long,
#                 "word_short": word_short,
#                 "syllable_long": syllable_long,
#                 "syllable_short": syllable_short,
#             }
#         )
#     return output
