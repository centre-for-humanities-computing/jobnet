"""Functions for preprocessing Danish and English text"""

import regex as re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import spacy

nltk.download("punkt")
from typing import List


def remove_html_commands(text: str) -> str:
    """
    Cleans text from html tags.

    Args:
        text (str): The string to remove tags from.

    Returns:
        str: A string cleaned from html tags.
    """

    soup = BeautifulSoup(text, features="html.parser")

    for script in soup(["script", "style"]):
        script.extract()

    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


def substitute_letter(text: str) -> str:
    """
    Substitutes Danish special letters with the respective equivalents.

    Args:
        text (str): The string to change letters in.

    Returns:
        str: The input string with Danish special letters substituted.
    """

    changed_char = re.sub("ø", "oe", text)
    changed_char = re.sub("æ", "ae", changed_char)
    changed_char = re.sub("å", "aa", changed_char)
    changed_char = re.sub("Ø", "oe", changed_char)
    changed_char = re.sub("Æ", "ae", changed_char)
    changed_char = re.sub("Å", "aa", changed_char)
    changed_char = re.sub("ü", "ue", changed_char)
    changed_char = re.sub("Ü", "ue", changed_char)
    changed_char = re.sub("ä", "ae", changed_char)
    changed_char = re.sub("Ä", "ae", changed_char)
    changed_char = re.sub("ö", "oe", changed_char)
    changed_char = re.sub("Ö", "oe", changed_char)

    return changed_char


def clean_text(text: str) -> str:
    """
    Cleans text from punctuation, URLs, special characters, multiple spaces and lowercases.

    Args:
        text (str): The string to clean.

    Returns:
        str: The cleaned string.
    """

    no_urls = re.sub(r"http\S+", "", text)
    no_special_ch = re.sub(
        r"(#[A-Za-z]+)|(@[A-Za-z]+)|([^A-Za-z \t])|(\w+:\/\/\S+)", " ", no_urls
    )
    no_special_ch = no_special_ch.replace("\n", " ")
    lowercased_str = no_special_ch.lower()
    cleaned_text = " ".join(lowercased_str.split())

    return cleaned_text


def collect_lemmas(text: str, nlp) -> List[str]:
    """
    Lemmatizes text using spaCy pipeline.

    Args:
        text (str): A string to be lemmatized.
        nlp: A spaCy pipeline.

    Returns:
        List[str]: A list with lemmas.
    """

    lemmas = []

    doc = nlp(text)
    for token in doc:
        lemmas.append(token.lemma_)

    return lemmas


def collect_nn_adj(text: str, nlp, pos_tags: List[str]) -> List[str]:
    """Collects lemmas only with a specified POS tag.

    Args:
        text (str): A text to extract lemmas from.
        nlp: A spaCy pipeline.
        pos_tags (_type_): A list with POS tags.

    Returns:
        List[str]: A list with lemmas.
    """

    doc = nlp(text)
    lemmas = []

    for token in doc:
        if token.pos_ in pos_tags:
            lemmas.append(token.lemma_)
    return lemmas


def rm_stops(text: List[str], stopwords: List[str]) -> List[str]:
    """
    Removes stopwords from tokenized/lemmatized text.

    Args:
        text (List[str]): A list with tokens/lemmas.
        stopwords (List[str]): A list of stopwords.

    Returns:
       List[str]: A list with tokens/lemmas without stopwords.
    """

    no_stopwords = []

    for word in text:
        if word not in stopwords:
            no_stopwords.append(word)

    return no_stopwords


def identity_tokenizer(text):
    return text
