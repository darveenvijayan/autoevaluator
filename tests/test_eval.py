from openai import OpenAI
from autoevaluator.eval import AutoEval, LLM_autoeval
from autoevaluator.client import setup_client
import pytest

@pytest.fixture
def client():
    client_, model_ =  setup_client()
    # Apply the patch to the OpenAI client
    yield client_

@pytest.fixture
def model():
    client_, model_ =  setup_client()
    yield model_

def test_llm_autoeval_all_true_positives(model, client):
    """Test when all claims are true positives."""
    claims = "The sky is blue. Water is wet."
    ground_truths = "The sky is blue. Water is wet."
    model_name = model  
    expected_results = {
        "TP": ["The sky is blue.", "Water is wet."],
        "FP": [],
        "FN": [],
        "recall": 1.0,
        "precision": 1.0,
        "f1_score": 1.0,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)
    assert actual_results == expected_results

# def test_llm_autoeval_with_false_positives(model, client):
#     """Test with some false positive claims."""
#     claims = "The sky is green. Birds can fly. Fish can breathe air."
#     ground_truths = "Birds can fly. The ocean is salty."
#     model_name = model
#     expected_results = {
#         "TP": ["Birds can fly."],
#         "FP": ["The sky is green.", "Fish can breathe air."],
#         "FN": ["The ocean is salty."],
#         "recall": 0.5,
#         "precision": 1/3,
#         "f1_score": 0.4,
#     }
#     actual_results = LLM_autoeval(claims, ground_truths, model_name, client)
#     assert actual_results == expected_results