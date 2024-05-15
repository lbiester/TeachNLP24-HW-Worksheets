import math

from model import NBLangIDModel


def main():
    """
    Use this test script to test your model on the "mini" data set, which includes
    the example from class.
    """
    # data (yes, these are single words, not actual sentences)
    train_sentences = ["ablaze", "hablo", "learn"]
    train_labels = ["eng", "spa", "eng"]
    test_sentences = ["able"]

    # train model and predict language for the one test sentence
    model = NBLangIDModel(ngram_size=2)
    model.fit(train_sentences, train_labels)
    results = model.predict_one_log_proba(test_sentences[0])

    # convert from log probabilities
    print({lang: math.e ** log_prob
           for lang, log_prob in results.items()})


if __name__ == "__main__":
    main()
