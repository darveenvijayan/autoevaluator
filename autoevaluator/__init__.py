from .client import setup_client
from .simplify import text_simplifier
from .eval import evaluate
from .eval_v2 import evaluate as evaluate_v2
from typing import Dict
from openai import OpenAI, AzureOpenAI