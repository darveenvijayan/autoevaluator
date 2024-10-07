from openai import OpenAI
from autoevaluator.eval import AutoEval, LLM_autoeval
from autoevaluator.client import setup_client
import pytest

@pytest.fixture
def client():
    client_, model_ =  setup_client()
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

def test_llm_autoeval_with_false_positives(model, client):
    """Test with some false positive claims."""
    claims = "The sky is green. Birds can fly. Fish can breathe air."
    ground_truths = "Birds can fly. The ocean is salty."
    model_name = model
    expected_results = {
        "TP": ["Birds can fly."],
        "FP": ["The sky is green.", "Fish can breathe air."],
        "FN": ["The ocean is salty."],
        "recall": 0.5,
        "precision": 1/3,
        "f1_score": 0.4,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

    assert actual_results['recall'] == expected_results['recall']
    assert actual_results['precision'] == expected_results['precision']
    assert actual_results['f1_score'] == expected_results['f1_score']

def test_llm_autoeval_with_historical_facts(model, client):
    """Test with some historical facts."""
    claims = "The moon landing was in 1969. The Great Wall of China is visible from space. Julius Caesar was a Roman emperor."
    ground_truths = "The moon landing was in 1969. Julius Caesar was a Roman general."
    model_name = model
    expected_results = {
        "TP": ["The moon landing was in 1969."],
        "FP": ["The Great Wall of China is visible from space.", "Julius Caesar was a Roman emperor."],
        "FN": ["Julius Caesar was a Roman general."],
        "recall": 0.5,
        "precision": 1/3,
        "f1_score": 0.4,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

    assert actual_results['recall'] == expected_results['recall']
    assert actual_results['precision'] == expected_results['precision']
    assert actual_results['f1_score'] == expected_results['f1_score']

def test_llm_autoeval_with_scientific_facts(model, client):
    """Test with some scientific facts."""
    claims = "Water boils at 100°C. Humans have 5 senses. The Earth is the center of the universe."
    ground_truths = "Water boils at 100°C. Humans have more than 5 senses."
    model_name = model
    expected_results = {
        "TP": ["Water boils at 100°C."],
        "FP": ["Humans have 5 senses.", "The Earth is the center of the universe."],
        "FN": ["Humans have more than 5 senses."],
        "recall": 0.5,
        "precision": 1/3,
        "f1_score": 0.4,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

    assert actual_results['recall'] == expected_results['recall']
    assert actual_results['precision'] == expected_results['precision']
    assert actual_results['f1_score'] == expected_results['f1_score']


def test_llm_autoeval_with_environmental_statements(model, client):
    """Test with some environmental statements."""
    claims = "Deforestation significantly contributes to climate change by reducing the number of trees that can absorb carbon dioxide from the atmosphere. The Amazon rainforest is often referred to as the lungs of the Earth due to its vast capacity to produce oxygen. Plastic pollution in the oceans is a minor issue that does not significantly impact marine life."
    ground_truths = "Deforestation significantly contributes to climate change by reducing the number of trees that can absorb carbon dioxide from the atmosphere. The Amazon rainforest is often referred to as the lungs of the Earth due to its vast capacity to produce oxygen."
    model_name = model
    expected_results = {
        "TP": ["Deforestation significantly contributes to climate change by reducing the number of trees that can absorb carbon dioxide from the atmosphere.", "The Amazon rainforest is often referred to as the lungs of the Earth due to its vast capacity to produce oxygen."],
        "FP": ["Plastic pollution in the oceans is a minor issue that does not significantly impact marine life."],
        "FN": [],
        "recall": 1.0,
        "precision": 2/3,
        "f1_score": 0.8,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

    assert actual_results['recall'] == expected_results['recall']
    assert actual_results['precision'] == expected_results['precision']
    assert actual_results['f1_score'] == expected_results['f1_score']

def test_llm_autoeval_with_health_nutrition_statements(model, client):
    """Test with some health and nutrition statements."""
    claims = "A balanced diet that includes a variety of fruits and vegetables can help reduce the risk of chronic diseases such as heart disease and diabetes. Regular physical activity is essential for maintaining a healthy weight and improving overall cardiovascular health. Consuming large amounts of sugar has no significant impact on one's health."
    ground_truths = "A balanced diet that includes a variety of fruits and vegetables can help reduce the risk of chronic diseases such as heart disease and diabetes. Regular physical activity is essential for maintaining a healthy weight and improving overall cardiovascular health."
    model_name = model
    expected_results = {
        "TP": ["A balanced diet that includes a variety of fruits and vegetables can help reduce the risk of chronic diseases such as heart disease and diabetes.", "Regular physical activity is essential for maintaining a healthy weight and improving overall cardiovascular health."],
        "FP": ["Consuming large amounts of sugar has no significant impact on one's health."],
        "FN": [],
        "recall": 1.0,
        "precision": 2/3,
        "f1_score": 0.8,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

    assert actual_results['recall'] == expected_results['recall']
    assert actual_results['precision'] == expected_results['precision']
    assert actual_results['f1_score'] == expected_results['f1_score']

# def test_llm_autoeval_with_technological_advancements(model, client):
#     """Test with some technological advancements."""
#     claims = "Artificial intelligence has the potential to revolutionize many industries by automating complex tasks and providing insights through data analysis. Quantum computing, which leverages the principles of quantum mechanics, promises to solve problems that are currently intractable for classical computers. The internet was invented in the 1980s and has since become an integral part of modern life, connecting people and information globally."
#     ground_truths = "Artificial intelligence has the potential to revolutionize many industries by automating complex tasks and providing insights through data analysis. Quantum computing, which leverages the principles of quantum mechanics, promises to solve problems that are currently intractable for classical computers."
#     model_name = model
#     expected_results = {
#         "TP": ["Artificial intelligence has the potential to revolutionize many industries by automating complex tasks and providing insights through data analysis.", "Quantum computing, which leverages the principles of quantum mechanics, promises to solve problems that are currently intractable for classical computers."],
#         "FP": ["The internet was invented in the 1980s and has since become an integral part of modern life, connecting people and information globally."],
#         "FN": [],
#         "recall": 1.0,
#         "precision": 2/3,
#         "f1_score": 0.8,
#     }
#     actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

#     assert actual_results['recall'] == expected_results['recall']
#     assert actual_results['precision'] == expected_results['precision']
#     assert actual_results['f1_score'] == expected_results['f1_score']

def test_llm_autoeval_with_economic_statements(model, client):
    """Test with some economic statements."""
    claims = "Inflation occurs when the general price level of goods and services rises, eroding purchasing power. High unemployment rates can lead to decreased consumer spending and slow economic growth. A country's GDP is the total value of all goods and services produced within its borders in a given year, and it is the only measure of economic health."
    ground_truths = "Inflation occurs when the general price level of goods and services rises, eroding purchasing power. High unemployment rates can lead to decreased consumer spending and slow economic growth."
    model_name = model
    expected_results = {
        "TP": ["Inflation occurs when the general price level of goods and services rises, eroding purchasing power.", "High unemployment rates can lead to decreased consumer spending and slow economic growth."],
        "FP": ["A country's GDP is the total value of all goods and services produced within its borders in a given year, and it is the only measure of economic health."],
        "FN": [],
        "recall": 1.0,
        "precision": 2/3,
        "f1_score": 0.8,
    }
    actual_results = LLM_autoeval(claims, ground_truths, model_name, client)

    assert actual_results['recall'] == expected_results['recall']
    assert actual_results['precision'] == expected_results['precision']
    assert actual_results['f1_score'] == expected_results['f1_score']
