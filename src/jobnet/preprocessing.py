"""Functions for preprocessing Danish and English text"""

import regex as re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
import spacy

nltk.download("punkt")
from typing import List
import pandas as pd


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


def clean_text(text: str) -> str:
    """
    Cleans text from punctuation, URLs, special characters, multiple spaces and lowercases.

    Args:
        text (str): The string to clean.

    Returns:
        str: The cleaned string.
    """

    no_urls = re.sub(r"http\S+", "", text)
    no_special_ch = re.sub(r"([^A-Za-zØøÅåÆæ])|(\w+:\/\/\S+)", " ", no_urls)
    no_special_ch = no_special_ch.replace("\n", " ")
    lowercased_str = no_special_ch.lower()
    cleaned_text = " ".join(lowercased_str.split())

    return cleaned_text


def collect_tokens(text: str, nlp) -> List[str]:
    """Collects tokens from a text.

    Args:
        text (str): A text to tokenize.
        nlp : A spaCy pipeline.

    Returns:
        List[str]: A list with tokens.
    """

    doc = nlp(text)
    tokens = []

    for token in doc:
        tokens.append(token.text)

    return tokens


def collect_lemmas(text: str, nlp) -> List[str]:
    """
    Lemmatizes text using spaCy pipeline.

    Args:
        text (str): A text to extract lemmas from.
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
    """
    Collects lemmas only with a specified POS tag.

    Args:
        text (str): A text to extract lemmas from.
        nlp: A spaCy pipeline.
        pos_tags (List[str]): A list with POS tags.

    Returns:
        List[str]: A list with lemmas.
    """

    nn_adj = []

    doc = nlp(text)

    for token in doc:
        if token.pos_ in pos_tags:
            nn_adj.append(token.lemma_)
    return nn_adj


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


def create_dfs_list(
    df: pd.DataFrame, lemmas: str, occupation_areas: List[str]
) -> List[pd.Series]:
    """Creates a list of pandas Serieses.

    Args:
        df (pd.Dataframe): The dataframe with preprocessed job posts.
        lemmas (str): The name of a Series column with lemmatised job posts.
        occupation_areas (List[str]): List of occupation areas names.

    Returns:
        List[pd.Series: The list of pandas Serieses per occupation area.
    """

    dfs = []

    for area in occupation_areas:
        df0 = df[df["occupation_area"] == area]
        sr_lemmas = df0[lemmas]
        sr_lemmas = sr_lemmas.rename(area)
        dfs.append(sr_lemmas)
    dfs.append(df[lemmas].rename("all_data"))
    return dfs
