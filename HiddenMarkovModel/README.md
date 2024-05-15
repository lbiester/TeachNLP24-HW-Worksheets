# Assignment 2: Part-of-Speech Tagging with Hidden Markov Models

## Distributed Files
### Starter Code
The following files are edited by students as part of the assignment:
* `model.py`: students implement the `_predict_one` method and edit the `__init__` method in the `HMMPOSTagger` class.

The following files are used for testing:
* `test.py`: trains the `HMMPOSTagger`, as well as the baseline `BaselinePOSTagger` on `POS_train.txt` and tests them on `POS_dev.txt`. Outputs tag-level accuracy.
* `test_mini.py`: sets the probabilities to the values from the worksheet and tests the model on the example from the worksheet. Outputs the predicted tag sequence.

`error_helper.py` is included to assist students with error analysis in their report.

### Data
Two data files are available, `POS_train.txt` and `POS_dev.txt`. These files are generated based on the [Universal Dependencies GUM Dataset](https://github.com/UniversalDependencies/UD_English-GUMs), which was distributed under a Creative Commons license. The train set includes 8548 sentences and the dev set includes 1117 sentences. They have been pre-processed from the .`conllu` format in which they are distributed to a simple format of sentences separated by newlines in the format

```text
tok_1/tag_1 tok_2/tag_2 ... tok_n/tag_n
```

The first five lines of `POS_train.txt` are provided below as a reference:
```txt
Aesthetic/ADJ Appreciation/NOUN and/CCONJ Spanish/ADJ Art/NOUN :/PUNCT
Insights/NOUN from/ADP Eye/NOUN -/PUNCT Tracking/NOUN
Claire/PROPN Bailey/PROPN -/PUNCT Ross/PROPN claire.bailey-ross@port.ac.uk/PROPN University/PROPN of/ADP Portsmouth/PROPN ,/PUNCT United/VERB Kingdom/PROPN
Andrew/PROPN Beresford/PROPN a.m.beresford@durham.ac.uk/PROPN Durham/PROPN University/PROPN ,/PUNCT United/VERB Kingdom/PROPN
Daniel/PROPN Smith/PROPN daniel.smith2@durham.ac.uk/PROPN Durham/PROPN University/PROPN ,/PUNCT United/VERB Kingdom/PROPN
```

## Undistributed Files
In addition to the autograder, the `POS_test.txt` file is kept in the private repository so that students do not have access to it to "game" the leaderboard.

The **starter code** for this assignment is redacted, as some faculty may choose to have students implement the estimation of probabilities and smoothing for this assignment. It can be requested by emailing lbiester@middlebury.edu.

## Additional Information
### Additional Homework Components
After implementing the Viterbi algorithm, students are asked to experiment with smoothing parameters such that they exceed the tag-level accuracy of a baseline model.

### Homework Report
In addition to writing code, all three assignments require that students write a short report. The report for this assignment requires students to:
* Report the tag-level accuracy of their model.
* Describe in plain English how they handle unknown words in their model.
* Perform error analysis for their model.

The report template in markdown format is available in `report.md`.

### Leaderboard
The first and second assignments both have an optional extension in which students try to improve their algorithm to compete on a gradescope leaderboard. In this assignment, student's model is trained on the training set and tested on a held out test set. Students are ranked according to tag-level accuracy (first) and speed (as a tiebreaker).
