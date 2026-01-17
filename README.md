# AutoEvaluator: LLM-Based Evaluation Framework

[![PyPI version](https://img.shields.io/pypi/v/autoevaluator.svg)](https://pypi.python.org/pypi/autoevaluator)
[![Python Version](https://img.shields.io/pypi/pyversions/autoevaluator.svg)](https://pypi.python.org/pypi/autoevaluator)
[![Downloads](https://img.shields.io/pypi/dm/autoevaluator)](https://pypi.python.org/pypi/autoevaluator)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Darveen_Vijayan-blue?link=https://www.linkedin.com/in/darveenvijayan)](https://www.linkedin.com/in/darveenvijayan)
[![Medium](https://img.shields.io/badge/Medium-LLMs%3A%20A%20Calculator%20for%20Words-green?link=https%3A%2F%2Fmedium.com%2Fthe-modern-scientist%2Flarge-language-models-a-calculator-for-words-7ab4099d0cc9)](https://medium.com/the-modern-scientist/large-language-models-a-calculator-for-words-7ab4099d0cc9)
[![Twitter Follow](https://img.shields.io/twitter/follow/DarveenVijayan?style=social)](https://twitter.com/DarveenVijayan)

AutoEvaluator is a powerful Python library that accelerates LLM output quality control through automated evaluation. Using LLMs to evaluate LLMs, it provides a simple, transparent, and developer-friendly API to identify True Positives (TP), False Positives (FP), and False Negatives (FN) in generated content against ground truth.

## 🚀 Features

- **Automated Evaluation**: Compare LLM outputs against ground truth with precision
- **Multi-Provider Support**: Works with AWS Bedrock, OpenAI, Anthropic, and Google Gemini
- **Comprehensive Metrics**: Automatically calculates Precision, Recall, and F1 Score
- **Async-First Design**: Built for high-performance concurrent evaluations
- **Structured Outputs**: Leverages Instructor for type-safe, validated responses
- **Sentence-Level Granularity**: Evaluates claims at the sentence level for detailed insights

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Supported Providers](#supported-providers)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

## 🔧 Installation

### Requirements

- Python 3.9 or higher
- An API key for at least one supported LLM provider

### Install via pip

```bash
pip install autoevaluator
```

### Install from source

```bash
git clone https://github.com/yourusername/autoevaluator.git
cd autoevaluator
pip install -e .
```

## ⚡ Quick Start

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load env variables BEFORE importing autoevaluator
from autoevaluator import evaluate, get_instructor_client

async def main():
    # Setup client for your preferred provider
    client = get_instructor_client(provider="openai", model="gpt-4o-mini")
    
    # Define the claim to evaluate
    claim = "Feynman was born in 1918 in Malaysia"
    
    # Define the ground truth
    ground_truth = "Feynman was born in 1918 in America."
    
    # Evaluate the claim
    result = await evaluate(
        claim=claim,
        ground_truth=ground_truth,
        client=client,
        model_name="gpt-4o-mini"
    )
    
    print(result)

# Run the async function
asyncio.run(main())
```

**Output:**

```python
{
    'TP': ['Feynman was born in 1918.'],
    'FP': ['Feynman was born in Malaysia.'],
    'FN': ['Feynman was born in America.'],
    'precision': 0.5,
    'recall': 0.5,
    'f1_score': 0.5
}
```

## 🔌 Supported Providers

AutoEvaluator supports multiple LLM providers out of the box:

| Provider | Models | Environment Variables |
|----------|--------|----------------------|
| **AWS Bedrock** | Claude Sonnet 4.5 | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` |
| **OpenAI** | GPT-4o, GPT-4o-mini, etc. | `OPENAI_API_KEY` |
| **Anthropic** | Claude Sonnet 4, etc. | `ANTHROPIC_API_KEY` |
| **Google Gemini** | Gemini 2.0 Flash, etc. | `GOOGLE_API_KEY` |

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# AWS Bedrock
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-southeast-1

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Google Gemini
GOOGLE_API_KEY=your_google_api_key
```

### Python Configuration

```python
import os

# Set environment variables programmatically
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
os.environ["AWS_ACCESS_KEY_ID"] = "your_aws_access_key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your_aws_secret_key"
```

## 💡 Usage Examples

### Example 1: Using OpenAI

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load env variables BEFORE importing autoevaluator
from autoevaluator import evaluate, get_instructor_client

async def evaluate_with_openai():
    client = get_instructor_client(provider="openai", model="gpt-4o-mini")
    
    claim = "The Earth is flat and the moon landing was in 1969."
    ground_truth = "The Earth is round. The moon landing was in 1969."
    
    result = await evaluate(claim, ground_truth, client=client, model_name="gpt-4o-mini")
    
    print(f"True Positives: {result['TP']}")
    print(f"False Positives: {result['FP']}")
    print(f"False Negatives: {result['FN']}")
    print(f"Precision: {result['precision']:.2f}")
    print(f"Recall: {result['recall']:.2f}")
    print(f"F1 Score: {result['f1_score']:.2f}")

asyncio.run(evaluate_with_openai())
```

### Example 2: Using AWS Bedrock

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load env variables BEFORE importing autoevaluator
from autoevaluator import evaluate, get_instructor_client

async def evaluate_with_bedrock():
    client = get_instructor_client(provider="bedrock")
    
    claim = "Python was created by Guido van Rossum in 1991."
    ground_truth = "Python was created by Guido van Rossum in 1991."
    
    result = await evaluate(claim, ground_truth, client=client, model_name="bedrock-claude")
    return result

result = asyncio.run(evaluate_with_bedrock())
print(f"Perfect match! F1 Score: {result['f1_score']}")
```

### Example 3: Using Anthropic

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load env variables BEFORE importing autoevaluator
from autoevaluator import evaluate, get_instructor_client

async def evaluate_with_anthropic():
    client = get_instructor_client(
        provider="anthropic",
        model="claude-sonnet-4-20250514"
    )
    
    claim = "Water boils at 100°C at sea level."
    ground_truth = "Water boils at 100°C at sea level."
    
    result = await evaluate(claim, ground_truth, client=client, model_name="claude-sonnet-4-20250514")
    return result

result = asyncio.run(evaluate_with_anthropic())
```

### Example 4: Batch Evaluation

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load env variables BEFORE importing autoevaluator
from autoevaluator import evaluate, get_instructor_client

async def batch_evaluate():
    client = get_instructor_client(provider="openai", model="gpt-4o-mini")
    
    test_cases = [
        {
            "claim": "Einstein developed the theory of relativity.",
            "ground_truth": "Einstein developed the theory of relativity."
        },
        {
            "claim": "The capital of France is London.",
            "ground_truth": "The capital of France is Paris."
        },
        {
            "claim": "Water is composed of hydrogen and oxygen.",
            "ground_truth": "Water is composed of hydrogen and oxygen."
        }
    ]
    
    tasks = [
        evaluate(tc["claim"], tc["ground_truth"], client=client, model_name="gpt-4o-mini")
        for tc in test_cases
    ]
    
    results = await asyncio.gather(*tasks)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"F1 Score: {result['f1_score']:.2f}")
        print(f"Precision: {result['precision']:.2f}")
        print(f"Recall: {result['recall']:.2f}")

asyncio.run(batch_evaluate())
```

## 📚 API Reference

### `evaluate()`

Evaluates a claim against ground truth and returns detailed metrics.

```python
async def evaluate(
    claim: str,
    ground_truth: str,
    client: instructor.AsyncInstructor,
    model_name: str = "gpt-4o-mini"
) -> Dict[str, Any]
```

**Parameters:**

- `claim` (str): The text to be evaluated
- `ground_truth` (str): The reference text to compare against
- `client` (instructor.AsyncInstructor): Instructor-wrapped async client
- `model_name` (str): Model identifier to use

**Returns:**

Dictionary containing:
- `TP` (List[str]): List of true positive sentences
- `FP` (List[str]): List of false positive sentences
- `FN` (List[str]): List of false negative sentences
- `precision` (float): Precision score (0.0 to 1.0)
- `recall` (float): Recall score (0.0 to 1.0)
- `f1_score` (float): F1 score (0.0 to 1.0)

### `get_instructor_client()`

Creates an Instructor-wrapped client for the specified LLM provider.

```python
def get_instructor_client(
    provider: Literal["bedrock", "openai", "anthropic", "gemini"] = "bedrock",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    mode: instructor.Mode = instructor.Mode.JSON,
    **kwargs
) -> instructor.AsyncInstructor
```

**Parameters:**

- `provider` (str): LLM provider to use ("bedrock", "openai", "anthropic", "gemini")
- `model` (Optional[str]): Model name (uses provider default if None)
- `api_key` (Optional[str]): API key (falls back to environment variables)
- `mode` (instructor.Mode): Instructor parsing mode
- `**kwargs`: Additional provider-specific arguments

**Returns:**

An Instructor-wrapped async client ready for use.

### `text_simplifier()`

Breaks down complex text into simple, single-clause sentences.

```python
async def text_simplifier(
    text: str,
    model_name: str,
    client: instructor.AsyncInstructor
) -> TextSimplify
```

## 🔍 How It Works

AutoEvaluator uses a sophisticated multi-step process to evaluate claims:

1. **Text Simplification**: Complex sentences are broken down into simple, atomic claims
2. **Question Generation**: Each simplified sentence is converted into a fact-checking question
3. **Bidirectional Verification**: Questions are checked against both the claim and ground truth
4. **Classification**: Sentences are classified as TP, FP, or FN based on verification results
5. **Metrics Calculation**: Precision, Recall, and F1 scores are computed from the classifications

### Architecture

```
Input Claim & Ground Truth
         ↓
   Text Simplifier (breaks into atomic sentences)
         ↓
   Question Generator (creates fact-check questions)
         ↓
   Question Checker (verifies against ground truth)
         ↓
   Classification (TP/FP/FN assignment)
         ↓
   Metrics Calculation (Precision, Recall, F1)
         ↓
   Structured Output
```

## 🎯 Advanced Usage

### Custom Text Simplification

```python
from autoevaluator import text_simplifier, get_instructor_client

async def simplify_text():
    client = get_instructor_client(provider="openai")
    
    complex_text = """Although the weather was bad and it was raining heavily, 
                      we decided to go hiking because we had planned it for weeks."""
    
    result = await text_simplifier(
        text=complex_text,
        model_name="gpt-4o-mini",
        client=client
    )
    
    print("Simplified sentences:")
    for sentence in result.simplified_sentences:
        print(f"- {sentence}")

asyncio.run(simplify_text())
```

### Using Provider-Specific Convenience Functions

```python
from autoevaluator.client import (
    get_openai_instructor_client,
    get_bedrock_instructor_client,
    get_anthropic_instructor_client,
    get_gemini_instructor_client
)

# OpenAI
openai_client = get_openai_instructor_client(model="gpt-4o")

# Bedrock
bedrock_client = get_bedrock_instructor_client()

# Anthropic
anthropic_client = get_anthropic_instructor_client()

# Gemini
gemini_client = get_gemini_instructor_client(model="gemini-2.0-flash")
```

### Error Handling

```python
import asyncio
from dotenv import load_dotenv
load_dotenv()  # Load env variables BEFORE importing autoevaluator
from autoevaluator import evaluate, get_instructor_client

async def safe_evaluate():
    try:
        client = get_instructor_client(provider="openai")
        result = await evaluate(
            claim="Some claim",
            ground_truth="Some truth",
            client=client,
            model_name="gpt-4o-mini"
        )
        return result
    except ValueError as e:
        print(f"Configuration error: {e}")
    except Exception as e:
        print(f"Evaluation error: {e}")

asyncio.run(safe_evaluate())
```

## 📊 Performance Considerations

- **Async by Default**: All operations are asynchronous for better performance
- **Batch Processing**: Use `asyncio.gather()` for concurrent evaluations
- **Rate Limiting**: Be mindful of provider rate limits when running batch evaluations
- **Caching**: Consider caching results for repeated evaluations

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Instructor](https://github.com/jxnl/instructor) for structured outputs
- Supports multiple LLM providers through unified interfaces
- Inspired by the need for automated, reliable LLM evaluation

## 📧 Contact

**Darveen Vijayan**

- LinkedIn: [darveenvijayan](https://www.linkedin.com/in/darveenvijayan)
- Twitter: [@DarveenVijayan](https://twitter.com/DarveenVijayan)
- Medium: [LLMs: A Calculator for Words](https://medium.com/the-modern-scientist/large-language-models-a-calculator-for-words-7ab4099d0cc9)

## 📈 Changelog

### Version 1.1.0
- Multi-provider support (OpenAI, Bedrock, Anthropic, Gemini)
- Async-first architecture
- Improved text simplification
- Enhanced error handling

---

**Made with ❤️ by [Darveen Vijayan](https://www.linkedin.com/in/darveenvijayan)**
