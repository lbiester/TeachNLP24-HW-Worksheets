# Assignment 3: Beam Search for Text Generation

## Distributed Files
### Starter Code
The following files are edited by students as part of the assignment:
* `decoding.py`: includes function signature for `generate_beam_search` which students implement. Code `generate_greedy` is redacted in case instructors would want to assign greedy search in addition to beam search, but is available upon request.

The following files are used for testing:
* `test_mini.py`: sets the probabilities to the values from the worksheet and tests the model on the example from the worksheet. Outputs the predicted tag sequence. Also includes a second example which was given in slides during class.

`util.py` includes the `next_token_probs` function, which abstracts away some of the process of generating a single token with GPT-2. `evaluate_bleu.py` is included to assist students with the report.

This assignment does not include a `test.py` file as no model is traned. However, students are encouraged to design their own test cases and experiment with different beam widths.

### Data
There is no data for the core part of this assignment. The starter code will automatically download GPT-2 weights through the Hugging Face transformer's library.

## Undistributed Files
The translated text and BLEU scores that are used in the report are not distributed, as they are not a core component of the programming assignment. They are available upon request.

### Worksheet Companion
This assignment includes `worksheet_companion.ipynb`, which students use to compute probabilities for the worksheet. It was originally distributed as a Colab notebeook.

## Additional Information
### Additional Worksheet & Homework Components
The back side of the worksheet asks students to compute modified n-gram precision to help them understand BLEU score.

### Homework Report
In addition to writing code, all three assignments require that students write a short report. The report for this assignment requires students to:
* Write one original test case where the output differs between beam search and greedy decoding. 
* Consider machine translated text using greedy decoding and beam search, and determine which translations are preferred. Compare their evaluation to the BLEU score.
  * Translations were available in Arabic, Chinese, French, German, Hebrew, Italian, Portuguese, Russian, and Spanish. A version using paraphrases was provided for unilingual students or students who did not speak one of the afformentioned languages. The languages were chosen because they were offered by the institution at which this assignment originated.

The report template in markdown format is available in `report.md`.
