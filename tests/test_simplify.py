from autoevaluator.simplify import text_simplifier, TextSimplify
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

def test_text_simplifier_basic(client, model):
    text = "This is a sentence"
    model_name = model
    result = text_simplifier(text, model_name, client)
    assert isinstance(result, TextSimplify)
    assert isinstance(result.simplified_sentences, list)
    assert len(result.simplified_sentences) > 0

def test_text_simplifier_empty_text(client, model):
    text = ""
    model_name = model
    result = text_simplifier(text, model_name, client)
    assert isinstance(result, TextSimplify)
    assert len(result.simplified_sentences) == 0

def test_text_simplifier_with_conjunctions(client, model):
    text = "This is a sentence with a conjunction, but it can be simplified."
    model_name = model
    result = text_simplifier(text, model_name, client)
    assert isinstance(result, TextSimplify)
    assert len(result.simplified_sentences) > 0 

def test_text_simplifier_with_commas(client, model):
    text = "This sentence has commas, which can make it complex, so we simplify it."
    model_name = model
    result = text_simplifier(text, model_name, client)
    assert isinstance(result, TextSimplify)
    assert len(result.simplified_sentences) > 0 

def test_text_simplifier_with_multiple_clauses(client, model):
    text = "This sentence has multiple clauses, and it's quite long, so we need to break it down."
    model_name = model
    result = text_simplifier(text, model_name, client)
    assert isinstance(result, TextSimplify)
    assert len(result.simplified_sentences) > 0 

def test_text_simplifier_with_different_model(client, model):
    text = "This is a sentence."
    model_name = model
    result = text_simplifier(text, model_name, client)
    assert isinstance(result, TextSimplify)
    assert len(result.simplified_sentences) > 0 

