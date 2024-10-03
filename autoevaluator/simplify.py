from pydantic import BaseModel, Field
from typing import List, Optional
from openai import OpenAI, AzureOpenAI
import instructor

class TextSimplify(BaseModel):
    simplified_sentences: List[str] = Field(
        ..., description="List of simplified sentences"
    )

def text_simplifier(text: str, model_name: str, client) -> TextSimplify:
    """Simplifies the given text into a list of single-clause sentences.

    Args:
        text (str): The text to simplify.
        model_name (str): The name of the language model to use for simplification.
        client (object): The client object used to interact with the language model.

    Returns:
        TextSimplify: A Pydantic model containing a list of simplified sentences.
    """
    completions = client.chat.completions.create(
        model=model_name,
        response_model=TextSimplify,
        messages=[
            {
                "role": "system",
                "content": "**You're an expert English language scholar!** "
                           "Your task is to break down this text into simple sentences "
                           "that each express a complete thought and contain only "
                           "a single independent clause. Split sentences with "
                           "conjunctions or commas."
            },
            {"role": "user", "content": f"text: {text}"},
        ],
    )
    return completions