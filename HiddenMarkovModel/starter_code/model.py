import abc
from typing import List, Tuple


PREDICTIONS_FILENAME = "predicted_tags.json"
UNK_TOKEN = "<UNK>"


def get_tokens(file_path: str) -> List[List[Tuple[str, str]]]:
    """
    Get the tokens and tags from a file.
    The file is expected to have sentences separated by newline.
    Each sentence is formatted as token1/tag1 token2/tag2 ... tokenn tagn

    Args:
        file_path (str): path to a file with the given format

    Returns:
        List[List[Tuple[str, str]]]: outer list represents sentences,
            inner list is tuples of (token, tag) pairs
    """
    sentences = []
    with open(file_path) as f:
        for line in f.readlines():
            tokens = []
            for token_tag_pair in line.split(" "):
                # str.rsplit("/", 1) returns splits once based on the last
                # occurrence of / this is important if the token has / in it
                token, tag = token_tag_pair.strip().rsplit("/", 1)
                tokens.append((token, tag))
            sentences.append(tokens)
    return sentences


class POSTagger(abc.ABC):
    """
    The starter code for this class is redacted so as to not trivialize similar
    assignments with less starter code provided.
    Please email lbiester@middlebury.edu for access to the starter code and
    share:
    * Your name
    * Your affiliation
    * The class that you are teaching
    """


class BaselinePOSTagger(POSTagger):
    """
    The starter code for this class is redacted so as to not trivialize similar
    assignments with less starter code provided.
    Please email lbiester@middlebury.edu for access to the starter code and
    share:
    * Your name
    * Your affiliation
    * The class that you are teaching
    """


class HMMPOSTagger(POSTagger):
    """
    The starter code for this class is redacted so as to not trivialize similar
    assignments with less starter code provided.
    Please email lbiester@middlebury.edu for access to the starter code and
    share:
    * Your name
    * Your affiliation
    * The class that you are teaching
    """