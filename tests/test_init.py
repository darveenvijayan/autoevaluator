from autoevaluator import evaluate, setup_client
import pytest

# Example ground truth and generated statement for testing
ground_truth = "This is a ground truth statement."
generated_statement = "This is a generated statement."

def test_evaluate_function():

    client, model_name =  setup_client()

    # Call the evaluate function
    eval_results = evaluate(generated_statement, ground_truth, client=client)

    # Assertions to check the output
    assert isinstance(eval_results, dict), "The output should be a dictionary"
    assert "TP" in eval_results, "The output dictionary should contain a 'TP' key"
    assert "FP" in eval_results, "The output dictionary should contain a 'FP' key"
    assert "FN" in eval_results, "The output dictionary should contain a 'FN' key"
    assert "precision" in eval_results, "The output dictionary should contain a 'precision' key"
    assert "recall" in eval_results, "The output dictionary should contain a 'recall' key"
    assert "f1_score" in eval_results, "The output dictionary should contain a 'f1_score' key"
