from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI, AzureOpenAI

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
        AutoEval: A Pydantic model containing classified statements (TP, FP, FN).
    """

    system_prompt = (
        "**You're an expert in logic and English. Your task is as follows:**\n"
        "  1. For each claim sentence, check if the information is supported by any of the truth sentences. If yes, it's a True Positive. If not, it's a False Positive.\n"
        "  2. For each truth sentence, check if the information is supported by any of the claim sentences. If yes, it's a False Negative.\n"
        "  Each sentence can only have one label (TP, FP, FN)."
    )

    completions = client.chat.completions.create(
        model=model_name,
        response_model=AutoEval,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CLAIMS: {claims}\nTRUTHS: {ground_truths}"},
        ],
    )
    return completions