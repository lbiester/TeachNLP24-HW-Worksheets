from typing import Dict, List

import torch
from transformers import PreTrainedModel


def next_token_probs(prompt_tokens: List[int], model: PreTrainedModel,  
                     n_tokens: int, log: bool = False) -> Dict[int, float]:
    """
    Get the n tokens with the highest probability along with their probability
    from a pytorch generation model

    Args:
        prompt_tokens (List[int]): the tokens from the prompt
        model (PreTrainedModel): the model
        n_tokens (int): the number of tokens to return
        log (bool): use log softmax. Defaults to False.

    Returns:
        Dict[int, float]: a dictionary mapping token IDs to probabilities
    """
    logits = model(torch.LongTensor(prompt_tokens)).logits
    # this computes the softmax of the logits.
    # the first index, 0, is for the batch
    # the second index, -1, is for the last token
    if log:
        next_token_probs = torch.log_softmax(logits[-1, :], 0)
    else:
        next_token_probs = torch.softmax(logits[-1, :], 0)
    # this sorts the probabilities and returns the INDICES of the probabilities
    # in descending sorted order
    sorted_token_ids = torch.argsort(next_token_probs, descending=True)
    top_token_ids = [tok_id.item() for tok_id in sorted_token_ids[:n_tokens]]
    # convert to a dictionary mapping token IDs to probabilities
    return dict(zip(top_token_ids, next_token_probs[top_token_ids].tolist()))