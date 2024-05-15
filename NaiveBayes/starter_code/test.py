import argparse
from scoring import accuracy_score, confusion_matrix
from util import LANGUAGES, load_data, print_confusion_matrix
from model import NBLangIDModel


def main():
    """
    Use this test script to test your model on a larger data set.
    To test on the full data set with character bigrams, run:
        python test.py data/train.tsv data/test.tsv

    Also see the descriptions of additional optional arguments below
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
    parser.add_argument(
        "--avg_samples_per_language",
        type=int,
        help="The number of samples to use per language. If not given, loads the full dataset. "
             "Set this to a number like 100 when debugging, then increase to increase model "
             "performance.")
    parser.add_argument(
        "--ngram_size",
        default=2,
        type=int,
        help="The size of character n-grams to use")
    args = parser.parse_args()

    # load data
    train_sentences, train_labels = load_data(
        args.train_file_path, avg_samples_per_language=args.avg_samples_per_language)
    test_sentences, test_labels = load_data(
        args.test_file_path, avg_samples_per_language=args.avg_samples_per_language)

    # train model and get predictions
    model = NBLangIDModel(ngram_size=args.ngram_size)
    model.fit(train_sentences, train_labels)
    predictions = model.predict(test_sentences)

    # evaluate model
    print(accuracy_score(test_labels, predictions))
    print_confusion_matrix(confusion_matrix(test_labels, predictions, LANGUAGES), LANGUAGES)


if __name__ == "__main__":
    main()
