import argparse
from model import BaselinePOSTagger, HMMPOSTagger


def main():
    """
    Train/test baseline POS tagger and a HMM
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_file_path",
        type=str,
        help="The file to use for training")
    parser.add_argument(
        "test_file_path",
        type=str,
        help="The file to use for testing")
    args = parser.parse_args()

    print("Baseline")
    print("--------------")
    tagger = BaselinePOSTagger()
    tagger.train(args.train_file_path)
    tagger.predict(args.test_file_path)
    print("\n\n")

    print("Hidden Markov Model")
    print("--------------")
    tagger = HMMPOSTagger()
    tagger.train(args.train_file_path)
    tagger.predict(args.test_file_path)


if __name__ == "__main__":
    main()
