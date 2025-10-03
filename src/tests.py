from src.helpers import get_words, get_syllables, get_clean_text
from src.models import WordsSyllables, TextWordsSyllables, WordsSyllablesComparison
import unittest


class TestGetCleanText(unittest.TestCase):
    def test_get_clean_text_remove_punctuation(self):
        input_text = """
        But “I know whom 'I' [have] (believed),
        =+-1234567890`~
        and am per-su-ad-ed that he/ is able
        to keep that which I have com,m.i<>tted
        unto him against that day!@#$%^&*|.”
        """
        output_text = get_clean_text(input_text)
        print(output_text)
        desired_output_text = """
        """
        self.assertEqual(output_text, desired_output_text)

    def test_get_clean_text_preserve_contractions(self):
        input_text = """I've I have He's he has He'll"""
        output_text = get_clean_text(input_text)
        desired_output_text = input_text
        self.assertEqual(output_text, desired_output_text)

    def test_get_clean_text_convert_apostrophes(self):
        input_text = f"""I've He{chr(39)}s He{chr(8217)}ll We{chr(8219)}ll they{chr(8216)}ll Ramses{chr(700)}s"""
        desired_output_text = f"""I've He's He'll We'll they'll Ramses's"""
        output_text = get_clean_text(input_text)
        self.assertEqual(output_text, desired_output_text)


class TestGetWords(unittest.TestCase):
    def test_get_clean_text_preserve_contractions(self):
        input_text = """I've got Ramses's 'sandals' and 'James's' flip flops and Moses' sneakers"""
        output_words = get_words(input_text)
        desired_output_words = [
            "I've",
            "Ramses's",
            "'sandals'",
            "'James's'",
            "Moses'",
            "sneakers",
        ]
        self.assertEqual(output_words, desired_output_words)

    def test_get_clean_text_preserve_contractions(self):
        input_text = """I've I have He's he has He'll"""
        output_text = get_clean_text(input_text)
        desired_output_text = input_text
        self.assertEqual(output_text, desired_output_text)

    def test_get_clean_text_convert_apostrophes(self):
        input_text = f"""I've He{chr(39)}s He{chr(8217)}ll We{chr(8219)}ll they{chr(8216)}ll Ramses{chr(700)}s"""
        desired_output_text = f"""I've He's He'll We'll they'll Ramses's"""
        output_text = get_clean_text(input_text)
        self.assertEqual(output_text, desired_output_text)
