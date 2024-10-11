from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from openai import OpenAI, AzureOpenAI
from .simplify import text_simplifier

class AutoEval(BaseModel):
    TP: List[str] = Field(..., description="List of True Positive statements")
    FP: List[str] = Field(..., description="List of False Positive statements")
    FN: List[str] = Field(..., description="List of False Negative statements")


def LLM_autoeval(claims: str, ground_truths: str, model_name: str, client: OpenAI | AzureOpenAI) -> AutoEval:
    """
    Performs automatic evaluation of claims against ground truths using a large language model.

    Args:
        claims (str): The text containing the claims to be evaluated.
        ground_truths (str): The text containing the ground truths.
        model_name (str): The name of the large language model to use.
        client (OpenAI | AzureOpenAI): An instance of the OpenAI or AzureOpenAI client.

    Returns:
        AutoEval: A Pydantic model containing classified statements (TP, FP, FN, recall, precision, f1_score).
    """

    system_prompt ="""You're an expert in logic and English. 

                        Your task is to assign labels to the given sentences. Think clearly, step by step.

                        Here are the rules you should follow:
                        1. For each CLAIMS sentence, if the sentence is supported by any of the TRUTHS sentences, it is labeled as True Positive (TP). 
                        2. For each CLAIMS sentence, if the sentence is NOT supported by any of the TRUTHS sentences, it is labeled as False Positive (FP).
                        3. For each TRUTHS sentence, if the sentence is NOT supported by any of the CLAIMS sentences, it is labeled as False Negative (FN).
                        
                        IMPORTANT: Each sentence can only have one label (TP / FP / FN)."""

    eval_results = client.chat.completions.create(
        model=model_name,
        response_model=AutoEval,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CLAIMS: {claims}\nTRUTHS: {ground_truths}"},
        ],
    ).dict()

    tp = len(eval_results['TP'])
    fp = len(eval_results['FP'])
    fn = len(eval_results['FN'])

    # Calculate recall, handling division by zero
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    eval_results['recall'] = recall
    # Calculate precision, handling division by zero
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    eval_results['precision'] = precision
    # Calculate F1-score, handling division by zero and avoiding redundant calculations
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    eval_results['f1_score'] = f1_score
    return eval_results

def evaluate(claim: str, ground_truth: str, client: OpenAI | AzureOpenAI, model_name: str = "gpt-4o-mini") -> Dict:
    """
    Evaluates a claim against a ground truth using a language model.

    Args:
        claim (str): The claim to be evaluated.
        ground_truth (str): The ground truth statement.
        model_name (str): The name of the language model to use.
        client (object): The client object used to interact with the language model.

    Returns:
        Dict: A dictionary containing the evaluation results (TP, FP, FN, recall, precision, f1_score).
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