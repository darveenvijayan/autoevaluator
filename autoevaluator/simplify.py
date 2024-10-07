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
    if text == "":
        return TextSimplify(simplified_sentences=[])
    completions = client.chat.completions.create(
        model=model_name,
        response_model=TextSimplify,
        messages=[
            {
                "role": "system",
                "content": f"""You're an expert in English language! You are also very detailed with your work.
                               Your task is to break down the TEXT from user into simple sentences.
                               Split sentences with conjunctions or commas into simpler sentences.
                               Each sentence must express a complete thought and contain only one independent clause. 
                               
                               IMPORTANT!
                                - Do not use replace nouns with pronouns.
                                - Always reuse the names and nouns from the TEXT from user for clarity.
                               """
            },
            {"role": "user", "content": f"TEXT: Although the weather forecast predicted heavy rain and strong winds, we decided to go hiking because we had already planned the trip for weeks, and we didn’t want to miss the opportunity to explore the beautiful trails and enjoy the breathtaking views that the mountains had to offer."},
            {"role": "assistant", "content": """["The weather forecast predicted heavy rain.","The weather forecast predicted strong winds.","The group decided to go hiking.","The group had already planned the trip for weeks.","The group didn't want to miss the opportunity to explore the beautiful trails.","The group wanted to enjoy the breathtaking views that the mountains had to offer."]"""},

            {"role": "user", "content": f"TEXT: Feynmann was born in 1918 in America."},
            {"role": "assistant", "content": """["Feynmann was born in 1918.", "Feynmann was born in America."]"""},
            
            {"role": "user", "content": f"TEXT: {text}"},
        ],
    )
    return completions
