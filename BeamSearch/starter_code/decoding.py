from transformers import PreTrainedModel, PreTrainedTokenizerFast

from util import next_token_probs

def generate_greedy(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, \
                    max_new_tokens: int) -> str:
    """
    Generate text using greedy decoding with no sampling

    Args:
        prompt (str): the prompt
        model (PreTrainedModel): the model
        tokenizer (PreTrainedTokenizerFast): the tokenizer
        max_new_tokens (int): the maximum new tokens to generate

    Returns:
        str: the generated text (including the prompt)
    """
    """
    The starter code for this function is redacted so as to not trivialize similar
    assignments with less starter code provided.
    Please email lbiester@middlebury.edu for access to the starter code and
    share:
    * Your name
    * Your affiliation
    * The class that you are teaching
    """


def generate_beam_search(prompt: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast, \
                         max_new_tokens: int, beam_width: int) -> str:
    """
    Generate text using beam search with no sampling

    Args:
        prompt (str): the prompt
        model (PreTrainedModel): the model
        tokenizer (PreTrainedTokenizerFast): the tokenizer
        max_new_tokens (int): the maximum new tokens to generate
        beam_width (int): the beam width

    Returns:
        str: the generated text (including the prompt)
    """
    raise NotImplementedError
