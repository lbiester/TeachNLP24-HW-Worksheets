from decoding import generate_beam_search, generate_greedy

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def main():
    """
    Train/test your beam search function on the basic examples we saw in class.
    """

    print("Please note that this script tests only the basic functionality of your code!")

    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # TESTS FOR PROMPT 1: "Vermont is" (from slides)
    prompt_1 = "Vermont is"
    # test for greedy decoding
    greedy_yours_1 = generate_greedy(prompt_1, model, tokenizer, 3)
    greedy_hf_1 = generator(
        prompt_1, do_sample=False, num_beams=1, max_new_tokens=3, 
        pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    print(f"Prompt 1 greedy\nExpected: {greedy_hf_1}\nActual: {greedy_yours_1}\n")
    # test for beam search
    beam_yours_1 = generate_beam_search(prompt_1, model, tokenizer, 3, 2)
    beam_hf_1 = generator(
        prompt_1, do_sample=False, num_beams=2, max_new_tokens=3, 
        pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    print(f"Prompt 1 beam\nExpected: {beam_hf_1}\nActual: {beam_yours_1}\n")



    # TESTS FOR PROMPT 2: "The sun is shining" (from worksheet)
    prompt_2 = "The sun is shining"
    # test for greedy decoding
    greedy_yours_2 = generate_greedy(prompt_2, model, tokenizer, 3)
    greedy_hf_2 = generator(
        prompt_2, do_sample=False, num_beams=1, max_new_tokens=3, 
        pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    print(f"Prompt 2 greedy\nExpected: {greedy_hf_2}\nActual: {greedy_yours_2}\n")
    # test for beam search
    beam_yours_2 = generate_beam_search(prompt_2, model, tokenizer, 3, 2)
    beam_hf_2 = generator(
        prompt_2, do_sample=False, num_beams=2, max_new_tokens=3, 
        pad_token_id=tokenizer.eos_token_id)[0]["generated_text"]
    print(f"Prompt 2 beam\nExpected: {beam_hf_2}\nActual: {beam_yours_2}\n")


if __name__ == "__main__":
    main()
