from .client import setup_client
from .simplify import text_simplifier
from .eval import LLM_autoeval
from typing import Dict
from openai import OpenAI, AzureOpenAI
import instructor

client, model_name =  setup_client()
# Apply the patch to the OpenAI client
client = instructor.from_openai(client)

def evaluate(claim: str, ground_truth: str) -> Dict:
    """
    Evaluates a claim against a ground truth using a language model.

    Args:
        claim (str): The claim to be evaluated.
        ground_truth (str): The ground truth statement.
        model_name (str): The name of the language model to use.
        client (object): The client object used to interact with the language model.

    Returns:
        Dict: A dictionary containing the evaluation results (TP, FP, FN).
    """

    # Simplify claim and ground truth
    simplified_claim = text_simplifier(claim, model_name, client).dict()['simplified_sentences']
    simplified_ground_truth = text_simplifier(ground_truth, model_name, client).dict()['simplified_sentences']

    # Join sentences into strings
    claim_string = '\n'.join(simplified_claim)
    ground_truth_string = '\n'.join(simplified_ground_truth)

    # Perform evaluation
    evaluation_results = LLM_autoeval(claim_string, ground_truth_string, model_name, client)

    return evaluation_results