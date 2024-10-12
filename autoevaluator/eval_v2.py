from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from openai import OpenAI, AzureOpenAI
from .simplify import text_simplifier

class QuestionAnswer(BaseModel):
    ans:str = Field(
        ..., description="Original sentence"
    )
    q:str = Field(
        ..., description="Generated question to fact check"
    )

class QuestionAnswerList(BaseModel):
    QA_list: List[QuestionAnswer] = Field(
        ..., description="List of sentences with the generated fact check question"
    )

def question_generator(text: List[str], client: OpenAI | AzureOpenAI, model_name: str = "gpt-4o-mini") -> QuestionAnswerList:
    completions = client.chat.completions.create(
        model=model_name,
        response_model=QuestionAnswerList,
        messages=[
            {
                "role": "system",
                "content": f"""You're an expert Fact Checker! You are also very detailed with your work.
                               Your task is to generate one comprehensive question for each sentence in SENTENCE_LIST.
                               The generated question should be answerable by Yes or No.
                               """
            },
            {"role": "user", "content": f"SENTENCE_LIST: {text}"},
        ],
    )
    return completions

# check question
class QuestionLabel(BaseModel):
    q : str = Field(
        ..., description="Question")
    
    label: bool = Field(
        ..., description="True if question can be answered by text correctly, else False")
    
class QuestionList(BaseModel):
    Q_list: List[QuestionLabel] = Field(
        ..., description="List of questions and labels")
    

def question_checker(question_list: List[str], text: str, client: OpenAI | AzureOpenAI, model_name: str = "gpt-4o-mini") -> QuestionList:
    completions = client.chat.completions.create(
        model=model_name,
        response_model=QuestionList,
        messages=[
            {
                "role": "system",
                "content": f"""You're an expert in English language! You are also very detailed with your work.
                               Check if each question in QUESTION_LIST can be answered by the ANSWER_TEXT correctly.
                               label True if question can be answered by text, else label False.
                               """
            },
            {"role": "user", "content": f"""QUESTION_LIST: {question_list}
                                            ANSWER_TEXT:{text}"""},
        ],
    )
    return completions

# run analysis
def get_classification(claim: str, ground_truth: str, client: OpenAI | AzureOpenAI, model_name: str = "gpt-4o-mini"):
    
    simplified_claim = text_simplifier(claim, model_name, client=client).dict()['simplified_sentences']

    q_gen = question_generator(text=simplified_claim, client=client).dict()


    # put the questions in a list
    question_list = [q['q'] for q in q_gen['QA_list']]

    # check the questions
    checks = question_checker(question_list = question_list, text = ground_truth, client=client).dict()

    # add labels from checks dict to q_gen dict based on question
    for i, qa in enumerate(q_gen['QA_list']):
        for check in checks['Q_list']:
            if qa['q'] == check['q']:
                qa['label'] = check['label']

    # extract the ans and label from q_gen into a new dict called results
    results = []
    for qa in q_gen['QA_list']:
        results.append({qa['ans']: qa['label']})

    return results

def evaluate(claim: str, ground_truth: str, client: OpenAI | AzureOpenAI, model_name: str = "gpt-4o-mini"):

    claim_results = get_classification(claim, ground_truth, client=client)

    # check claim result and replace True with TP and False with FP
    for claim_result in claim_results:
        for key, value in claim_result.items():
            if value == True:
                claim_result[key] = 'TP'
            else:
                claim_result[key] = 'FP'

    gt_results = get_classification(ground_truth, claim, client=client)

    # check claim result and replace True with TP and False with FP
    for gt_result in gt_results:
        for key, value in gt_result.items():
            if value == True:
                gt_result[key] = '_TP'
            else:
                gt_result[key] = 'FN'

    # combine the dict
    claim_results.extend(gt_results)
    combined_results= claim_results

    # create adictionary with keys TP, FP and FN and list of sentences as values
    TP_list = []
    FP_list = []
    FN_list = []

    for results in combined_results:
        for key, value in results.items():
            if value == 'TP':
                TP_list.append(key)
            elif value == 'FP':
                FP_list.append(key)
            elif value == 'FN':
                FN_list.append(key)

    result_dict = {
        'TP': TP_list,
        'FP': FP_list,
        'FN': FN_list
    }

    tp = len(result_dict['TP'])
    fp = len(result_dict['FP'])
    fn = len(result_dict['FN'])

    # Calculate recall, handling division by zero
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    result_dict['recall'] = recall
    # Calculate precision, handling division by zero
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    result_dict['precision'] = precision
    # Calculate F1-score, handling division by zero and avoiding redundant calculations
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    result_dict['f1_score'] = f1_score
    return result_dict
