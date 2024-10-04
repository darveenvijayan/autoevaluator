# AutoEvaluator: An LLM based LLM Evaluator

AutoEvaluator is a Python library that speeds up the large language models (LLMs) output generation QC work. It provides a simple, transparent, and user-friendly API to identify the True Positives (TP), False Positives (FP), and False Negatives (FN) statements based the generated statement and ground truth provided. Get ready to turbocharge your LLM evaluations!

[![Autoevaluator](https://img.shields.io/pypi/v/autoevaluator.svg)](https://pypi.python.org/pypi/autoevaluator)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/autoevaluator)](https://pypi.python.org/pypi/autoevaluator)

[![Static Badge](https://img.shields.io/badge/LinkedIn-Darveen_Vijayan-blue?link=https://www.linkedin.com/in/darveenvijayan)](https://www.linkedin.com/in/darveenvijayan)
[![Static Badge](https://img.shields.io/badge/Medium-LLMs%3A%20A%20Calculator%20for%20Words-green?link=https%3A%2F%2Fmedium.com%2Fthe-modern-scientist%2Flarge-language-models-a-calculator-for-words-7ab4099d0cc9)](https://medium.com/the-modern-scientist/large-language-models-a-calculator-for-words-7ab4099d0cc9)
[![Twitter Follow](https://img.shields.io/twitter/follow/DarveenVijayan?style=social)](https://twitter.com/DarveenVijayan)



**Features:**

* Evaluate LLM outputs against a reference dataset or human judgement.
* Generate TP, FP, and FN sentences based on ground truth provided
* Calculate Precision, Recall and F1 score


### Installation

Autoevaluator requires `Python 3.9` and several dependencies. You can install autoevaluator:

```bash
pip install autoevaluator
```

### Usage

1. **Prepare your data:**
    * Create a dataset containing LLM outputs and their corresponding ground truth labels.
    * The format of the data can be customized depending on the evaluation task.
    * Example: A CSV file with columns for "prompt," "llm_output," and "ground_truth"

2. **setup environment variables**
```
import os
os.environ["OPENAI_API_KEY"] = "<OPENAI_API_KEY>"
os.environ["AZURE_OPENAI_API_KEY"] = "<AZURE_OPENAI_API_KEY>"
os.environ["AZURE_OPENAI_ENDPOINT"] = "<AZURE_OPENAI_ENDPOINT>"
os.environ["DEPLOYMENT"] = "<azure>/<not-azure>"
```

3. **run autoevaluator**
```
from autoevaluator import evaluate
eval_results = evaluate(generated_statement, ground_truth)
```

3. **Output:**
    * The script will generate a dictionary with the following information:
        * TP, FP, and FN sentences
        * Precision, Recall and F1 score

**License:**

This project is licensed under the MIT License. See the `LICENSE` file for details.
