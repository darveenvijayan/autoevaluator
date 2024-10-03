# autoevaluator

## LLM Evaluation Toolkit

This repository provides a toolkit for evaluating Large Language Model (LLM) outputs. It calculates and reports True Positives (TP), False Positives (FP), and False Negatives (FN).

**Features:**

* Evaluate LLM outputs against a reference dataset or human judgement.
* Calculate TP, FP, and FN scores for various evaluation tasks.


### Installation

autoevaluator requires Python 3.9 and several dependencies. You can install autoevaluator:

```bash
pip install git+https://github.com/darveenvijayan/autoevaluator.git
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
eval_results = evaluate(claim, ground_truth)
```

3. **Output:**
    * The script will generate a dictionary with the following information:
        * TP, FP, and FN sentences

**License:**

This project is licensed under the MIT License. See the `LICENSE` file for details.
