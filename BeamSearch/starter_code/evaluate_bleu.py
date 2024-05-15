import argparse
import random

import pandas as pd

LANGUAGE_TO_CODE = {
    "arabic": "ar", 
    "chinese": "zh", 
    "english": "en",
    "french": "fr",  
    "german": "de", 
    "hebrew": "he", 
    "italian": "it", 
    "portuguese": "pt", 
    "russian": "ru", 
    "spanish": "es" 
}


def get_preference(term: str) -> str:
    """
    Prompts a user to write their preference between translation 1 or 2 until the input is valid

    Args:
        term (str): translation or paraphrase, depending on the language

    Returns:
        str: the preferred translation ("1" or "2")
    """
    prompt = f"Do you prefer {term} 1 or {term} 2? "
    preferred = input(prompt)
    while preferred.lower().strip() not in ["1", "2", f"{term} 1", f"{term} 2"]:
        print("Please make sure that you are inputting 1 or 2.")
        preferred = input(prompt)
    return preferred[-1]


def create_summary_text(i: int, example: pd.Series, preferred_translation: str, term: str) -> str:
    """
    Create a summary of the results for example i

    Args:
        i (int): the index of the example
        example (pd.Series): the data for this example
        preferred_translation (str): the user's preferred translation
        term (str): translation or paraphrase, depending on the language

    Returns:
        str: the summary
    """
    summary = f"Example {i}\n=========\n"
    summary += f"Source: {example.source}\n"
    summary += f"Greedy {term}: {example.hypothesis_greedy}\n"
    summary += f"Greedy BLEU Score: {example.bleu_greedy}\n"
    summary += f"Beam Search {term}: {example.hypothesis_beam}\n"
    summary += f"Beam Search BLEU Score: {example.bleu_beam}\n"
    if term == "translation":
        summary += f"Reference Text: {example.targets}\n"

    if preferred_translation == "beam":
        summary += f"You preferred the beam search {term}\n\n"
    else:
        summary += f"You preferred the greedy {term}\n\n"
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", 
                        type=str.lower,
                        choices=["arabic", "chinese", "english", "french", "german", "hebrew", 
                                 "italian", "portuguese", "russian", "spanish"])
    args = parser.parse_args()

    lang_code = LANGUAGE_TO_CODE[args.language.lower()]

    # read results from Machine translation
    if args.language == "english":
        file_path = "/home/lbiester/CS457HW/HW6/data/paraphrase-en-ar-backtranslation.tsv"
        df = pd.read_csv(file_path, sep="\t")
        unique = df.apply(lambda row: len({row["source"].strip(), row["hypothesis_greedy"].strip(), row["hypothesis_beam"].strip()}) == 3, axis=1)
        examples = df[unique].sample(3).reset_index(drop=True)
    else:
        file_path = f"/home/lbiester/CS457HW/HW6/data/mt_results-{lang_code}-10-4-None.tsv"
        df = pd.read_csv(file_path, sep="\t")
        examples = df[df["hypothesis_greedy"] != df["hypothesis_beam"]].sample(3).reset_index(drop=True)

    term = "paraphrase" if args.language == "english" else "translation"

    summary = ""
    for i, example in examples.iterrows():
        print("Source sentence:", example.source)
        
        # Randomly choose which translation to show first, so that you don't know which is which
        beam_first = random.choice([True, False])

        if beam_first:
            print(f"{term} 1:", example.hypothesis_beam)
            print(f"{term} 2:", example.hypothesis_greedy)
        else:
            print(f"{term} 1:", example.hypothesis_greedy)
            print(f"{term} 2:", example.hypothesis_beam)

        preference = get_preference(term)

        # Find the preferred translation
        if (beam_first and preference == "1") or (not beam_first and preference == "2"):
            preferred_translation = "beam"
        else:
            preferred_translation = "greedy"

        # Add this example to summary
        summary += create_summary_text(i, example, preferred_translation, term)

        print("\n")

    print("EVALUATION SUMMARY")
    print(f"Your language of choice: {args.language}")
    print(summary)


if __name__ == "__main__":
    main()
