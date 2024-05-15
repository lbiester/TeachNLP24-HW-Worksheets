import json
import os
from random import shuffle
from typing import List

from model import PREDICTIONS_FILENAME


PRINT_NUM = 5


def print_predictions(tokens: List[str], golden_tags: List[str],
                      predicted_tags: List[str]):
    """
    Print the example error in the format used in the report

    Args:
        tokens (List[str]): tokens in the sentence
        golden_tags (List[str]): the actual POS tags
        predicted_tags (List[str]): the predicted POS tags
    """
    print("* Tokens:", " ".join(tokens))
    print("* Actual tags:", " ".join(golden_tags))
    print("* Predicted tags:", " ".join(predicted_tags))
    print("\n")


def main():
    """
    Read predictions file and print 5 random incorrect predictions
    """
    if os.path.exists(PREDICTIONS_FILENAME):
        with open(PREDICTIONS_FILENAME, "r") as f:
            predictions = json.load(f)
        shuffle(predictions)
        for tokens, golden_tags, predicted_tags in predictions[:PRINT_NUM]:
            print_predictions(tokens, golden_tags, predicted_tags)
    else:
        print(f"No {PREDICTIONS_FILENAME} file - have you run test.py?")


if __name__ == "__main__":
    main()
