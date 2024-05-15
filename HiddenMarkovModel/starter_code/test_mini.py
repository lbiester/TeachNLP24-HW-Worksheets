from model import HMMPOSTagger


def main():
    """
    Train/test your model WITHOUT SMOOTHING on the example from the worksheet.
    """

    print("Please note that this script tests the basic functionality of your code with no smoothing on a very small test set!")

    tagger = HMMPOSTagger(k_emission=0, k_transition=0)
    # instead of training the model, this test script will use the set_prob method
    # IF you want to test your model using this, you' ll need to implement that method
    # otherwise, you may skip it, but I do think that this will be helpful for debugging!
    # it tests the example from the worksheet

    print("Example: 'ski on snow'")

    # the initial, transition and emission log probabilities in the "bear is on the move" example
    init_log_probs = {"V": -3, "N": -3, "P": -4}

    transition_log_probs = {
        "V": {"V": -4, "N": -2, "P": -2},
        "N": {"V": -3, "N": -2, "P": -1},
        "P": {"V": -5, "N": -2, "P": -4}
    }

    emission_log_probs = {
        "V": {"ski": -6, "on": float("-inf"), "snow": -5},
        "N": {"ski": -5, "on": float("-inf"), "snow": -3},
        "P": {"ski": float("-inf"), "on": -1}, "snow": float("-inf")
    }

    tagger.set_model_params(init_log_probs, transition_log_probs, emission_log_probs)
    predictions = tagger.predict_one(["ski", "on", "snow"])
    print("Should be N P N:", predictions)


if __name__ == "__main__":
    main()
