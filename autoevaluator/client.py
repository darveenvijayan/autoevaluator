"""
Multi-Provider Client Factory with Instructor Support

This module provides a unified interface to create instructor-wrapped clients
for multiple LLM providers:
- AWS Bedrock (Claude via Bedrock)
- OpenAI (GPT models)
- Anthropic (Claude API)
- Google (Gemini models)

All clients are wrapped with Instructor for structured outputs and type-safe
AI responses using Pydantic models.

Usage:
    >>> from client import get_instructor_client
    >>> from pydantic import BaseModel
    >>> 
    >>> # Get a client for any provider
    >>> client = get_instructor_client(provider="openai", model="gpt-4o-mini")
    >>> 
    >>> # Define your response model
    >>> class Person(BaseModel):
    >>>     name: str
    >>>     age: int
    >>> 
    >>> # Get structured response
    >>> person = await client.chat.completions.create(
    >>>     model="gpt-4o-mini",
    >>>     response_model=Person,
    >>>     messages=[{"role": "user", "content": "Extract: John is 30 years old"}]
    >>> )
    >>> print(person.name, person.age)
"""

import instructor
from .LLM import (
    BedrockAsyncOpenAI,
    OpenAIAsyncClient,
    AnthropicAsyncOpenAI,
    GeminiAsyncOpenAI
)
from typing import Literal, Optional


def get_instructor_client(
    provider: Literal["bedrock", "openai", "anthropic", "gemini"] = "bedrock",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    mode: instructor.Mode = instructor.Mode.JSON,
    **kwargs
):
    """
    Create an Instructor-wrapped client for any supported LLM provider.
    
    This function creates a client for the specified provider and wraps it with
    Instructor, enabling type-safe structured outputs using Pydantic models.
    
    Args:
        provider: The LLM provider to use. Options:
            - "bedrock": AWS Bedrock (Claude via Bedrock)
            - "openai": OpenAI (GPT models)
            - "anthropic": Anthropic (Claude API)
            - "gemini": Google (Gemini models)
        model: The model to use. If None, uses provider defaults:
            - bedrock: "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
            - openai: "gpt-4o-mini"
            - anthropic: "claude-sonnet-4-20250514"
            - gemini: "gemini-2.0-flash"
        api_key: API key for the provider (defaults to env vars)
        mode: Instructor mode for structured output parsing
            Options: instructor.Mode.JSON, instructor.Mode.TOOLS, etc.
        **kwargs: Additional arguments passed to the provider client
    
    Returns:
        instructor.AsyncInstructor: An instructor-wrapped client
    
    Raises:
        ValueError: If an unsupported provider is specified
    
    Example:
        ```python
        from pydantic import BaseModel
        from client import get_instructor_client
        
        # Define response structure
        class UserInfo(BaseModel):
            name: str
            age: int
            email: str
        
        # Create client for OpenAI
        client = get_instructor_client(provider="openai", model="gpt-4o-mini")
        
        # Get structured response
        user = await client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=UserInfo,
            messages=[
                {"role": "user", "content": "Extract: Alice is 25, email: alice@example.com"}
            ]
        )
        
        print(f"Name: {user.name}, Age: {user.age}")
        ```
    """
    # Create the base client based on provider
    if provider == "bedrock":
        default_model = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
        base_client = BedrockAsyncOpenAI(
            model=model or default_model,
            **kwargs
        )
    
    elif provider == "openai":
        default_model = "gpt-4o-mini"
        base_client = OpenAIAsyncClient(
            model=model or default_model,
            api_key=api_key,
            **kwargs
        )
    
    elif provider == "anthropic":
        default_model = "claude-sonnet-4-20250514"
        base_client = AnthropicAsyncOpenAI(
            model=model or default_model,
            api_key=api_key,
            **kwargs
        )
    
    elif provider == "gemini":
        default_model = "gemini-2.0-flash"
        base_client = GeminiAsyncOpenAI(
            model=model or default_model,
            api_key=api_key,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers are: bedrock, openai, anthropic, gemini"
        )
    
    # Wrap with Instructor for structured outputs
    instructor_client = instructor.patch(
        base_client,
        mode=mode
    )
    
    return instructor_client


# Convenience functions for specific providers
def get_bedrock_instructor_client(
    model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    mode: instructor.Mode = instructor.Mode.JSON,
    **kwargs
):
    """
    Create an Instructor-wrapped Bedrock client.
    
    This is a convenience function that calls get_instructor_client with provider="bedrock".
    
    Args:
        model: Bedrock model ID (default: Claude Sonnet 4.5)
        mode: Instructor mode
        **kwargs: Additional arguments
    
    Returns:
        instructor.AsyncInstructor: An instructor-wrapped Bedrock client
    """
    return get_instructor_client(provider="bedrock", model=model, mode=mode, **kwargs)


def get_openai_instructor_client(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    mode: instructor.Mode = instructor.Mode.JSON,
    **kwargs
):
    """
    Create an Instructor-wrapped OpenAI client.
    
    This is a convenience function that calls get_instructor_client with provider="openai".
    
    Args:
        model: OpenAI model name (default: gpt-4o-mini)
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
        mode: Instructor mode
        **kwargs: Additional arguments
    
    Returns:
        instructor.AsyncInstructor: An instructor-wrapped OpenAI client
    """
    return get_instructor_client(provider="openai", model=model, api_key=api_key, mode=mode, **kwargs)


def get_anthropic_instructor_client(
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    mode: instructor.Mode = instructor.Mode.JSON,
    **kwargs
):
    """
    Create an Instructor-wrapped Anthropic client.
    
    This is a convenience function that calls get_instructor_client with provider="anthropic".
    
    Args:
        model: Claude model name (default: claude-sonnet-4-20250514)
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        mode: Instructor mode
        **kwargs: Additional arguments
    
    Returns:
        instructor.AsyncInstructor: An instructor-wrapped Anthropic client
    """
    return get_instructor_client(provider="anthropic", model=model, api_key=api_key, mode=mode, **kwargs)


def get_gemini_instructor_client(
    model: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
    mode: instructor.Mode = instructor.Mode.JSON,
    **kwargs
):
    """
    Create an Instructor-wrapped Gemini client.
    
    This is a convenience function that calls get_instructor_client with provider="gemini".
    
    Args:
        model: Gemini model name (default: gemini-2.0-flash)
        api_key: Google API key (defaults to GOOGLE_API_KEY env var)
        mode: Instructor mode
        **kwargs: Additional arguments
    
    Returns:
        instructor.AsyncInstructor: An instructor-wrapped Gemini client
    """
    return get_instructor_client(provider="gemini", model=model, api_key=api_key, mode=mode, **kwargs)


# Example usage for testing
if __name__ == "__main__":
    import asyncio
    from pydantic import BaseModel, Field
    
    class MovieReview(BaseModel):
        """Structure for a movie review"""
        title: str = Field(description="Movie title")
        rating: int = Field(description="Rating from 1-10")
        summary: str = Field(description="Brief review summary")
        would_recommend: bool = Field(description="Whether to recommend")
    
    async def test_provider(provider_name: str):
        """Test a specific provider"""
        print(f"\n{'='*60}")
        print(f"Testing {provider_name.upper()} Provider")
        print(f"{'='*60}")
        
        try:
            # Get the instructor client for the provider
            client = get_instructor_client(provider=provider_name)
            
            # Request structured output
            review = await client.chat.completions.create(
                response_model=MovieReview,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a movie review analyst. Extract structured information from reviews."
                    },
                    {
                        "role": "user",
                        "content": """Analyze this review:
                        
"The Matrix is an incredible sci-fi movie. I'd give it a 9/10.
The action sequences are groundbreaking and the story is mind-bending.
Definitely worth watching!"""
                    }
                ]
            )
            
            print(f"✓ Title: {review.title}")
            print(f"✓ Rating: {review.rating}/10")
            print(f"✓ Summary: {review.summary}")
            print(f"✓ Would Recommend: {review.would_recommend}")
            
        except Exception as e:
            print(f"✗ Error: {type(e).__name__}: {e}")
    
    async def run_all_tests():
        """Run tests for all providers"""
        print("\n" + "="*60)
        print("MULTI-PROVIDER INSTRUCTOR CLIENT TESTS")
        print("="*60)
        
        # Test each provider
        providers = ["bedrock", "openai", "anthropic", "gemini"]
        
        for provider in providers:
            await test_provider(provider)
        
        print("\n" + "="*60)
        print("All tests completed")
        print("="*60 + "\n")
    
    # Run the tests
    asyncio.run(run_all_tests())
