"""
Multi-Provider LLM Client Module

This module provides unified async client implementations for multiple LLM providers:
- AWS Bedrock (Claude via Bedrock)
- OpenAI (GPT models)
- Anthropic (Claude API)
- Google (Gemini models)

All clients inherit from openai.AsyncOpenAI to ensure compatibility with
ragas, Instructor, and other OpenAI-compatible frameworks.
"""

import os
import asyncio
import json
from dotenv import load_dotenv
import aioboto3
from botocore.exceptions import ClientError
from typing import Any, List, Optional, Dict
from pydantic import Field
import openai
from anthropic import AsyncAnthropic
from google import genai
from google.genai import types

load_dotenv()


# ============================================================================
# AWS BEDROCK IMPLEMENTATION
# ============================================================================

# Get AWS credentials from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")  # Optional
aws_region = os.getenv("AWS_REGION", "ap-southeast-1")


async def Bedrock_LLM(
    system_prompt: str = '',
    messages: list = None,
    temperature: float = 0.0,
    max_tokens: int = 4096
) -> str:
    """
    Call AWS Bedrock Claude model directly.
    
    Args:
        system_prompt: System prompt for the model
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum tokens in response
    
    Returns:
        Response text from the model
    """
    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your .env file"
        )
    
    # Set the model ID
    model_id = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # Format the request payload
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": messages
    }

    request = json.dumps(native_request)

    # Create an Amazon Bedrock Runtime client
    session = aioboto3.Session()
    
    async with session.client(
        service_name="bedrock-runtime",
        region_name=aws_region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    ) as brt:
        try:
            response = await brt.invoke_model(modelId=model_id, body=request)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            print(f"ERROR: Can't invoke '{model_id}'.")
            print(f"Error Code: {error_code}")
            print(f"Error Message: {error_message}")
            
            if error_code == 'UnrecognizedClientException':
                print("\nTroubleshooting steps:")
                print("1. Check if your AWS credentials are valid and not expired")
                print("2. If using temporary credentials (AWS SSO/STS), ensure AWS_SESSION_TOKEN is set")
                print("3. Verify you have the correct AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
                print("4. Run 'aws sts get-caller-identity' to verify your credentials work")
            
            raise
        except Exception as e:
            print(f"ERROR: Unexpected error invoking '{model_id}'. Reason: {e}")
            raise

        # Decode the response body
        response_body = await response["body"].read()
        model_response = json.loads(response_body)
        response_text = model_response["content"][0]["text"]

        return response_text


class BedrockAsyncOpenAI(openai.AsyncOpenAI):
    """
    AsyncOpenAI-compatible client that routes calls to AWS Bedrock.
    
    This class provides OpenAI API compatibility for AWS Bedrock Claude models.
    """

    def __init__(self, model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0", **kwargs):
        """
        Initialize the Bedrock client.
        
        Args:
            model: The Bedrock model ID to use
            **kwargs: Additional args (absorbed to prevent errors)
        """
        super().__init__(api_key="bedrock-placeholder", **kwargs)
        
        self.default_model = model
        self.chat = self.Chat(self)

    class Chat:
        """Chat interface that mimics openai.resources.chat.Chat"""
        
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            """Completions interface that mimics openai.resources.chat.Completions"""
            
            def __init__(self, parent):
                self.parent = parent
            
            async def create(
                self,
                model: str = None,
                messages: List[dict] = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
                response_model=None,
                **kwargs,
            ):
                """Create a chat completion."""
                model = model or self.parent.default_model
                
                # Extract system prompt and build message history
                system_prompt = ""
                bedrock_messages = []
                
                for msg in messages or []:
                    role = msg.get("role")
                    content = msg.get("content")
                    
                    if role == "system":
                        system_prompt = content
                    elif role in ["user", "assistant"]:
                        bedrock_messages.append({"role": role, "content": content})

                # Call Bedrock
                response_text = await Bedrock_LLM(
                    system_prompt=system_prompt,
                    messages=bedrock_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Create OpenAI-compatible response structure
                return self._create_response(response_text, model)
            
            def _create_response(self, text: str, model: str):
                """Create OpenAI-compatible response object."""
                class Message:
                    def __init__(self, content, role="assistant"):
                        self.content = content
                        self.role = role
                        self.function_call = None
                        self.tool_calls = None
                
                class Choice:
                    def __init__(self, message, index=0):
                        self.message = message
                        self.index = index
                        self.finish_reason = "stop"
                
                class Usage:
                    def __init__(self):
                        self.prompt_tokens = 0
                        self.completion_tokens = 0
                        self.total_tokens = 0
                
                class Response:
                    def __init__(self, text, model_used):
                        self.choices = [Choice(Message(text))]
                        self.model = model_used
                        self.id = "bedrock-completion"
                        self.object = "chat.completion"
                        self.created = 0
                        self.usage = Usage()
                    
                    def __getitem__(self, key):
                        return getattr(self, key)

                return Response(text, model)


# ============================================================================
# OPENAI IMPLEMENTATION
# ============================================================================

class OpenAIAsyncClient(openai.AsyncOpenAI):
    """
    Native OpenAI AsyncClient with optional default model.
    
    This is a thin wrapper around the official OpenAI client that allows
    setting a default model.
    """

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None, **kwargs):
        """
        Initialize the OpenAI client.
        
        Args:
            model: Default model to use (e.g., 'gpt-4o-mini', 'gpt-4o')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            **kwargs: Additional args passed to AsyncOpenAI
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or pass it explicitly"
            )
        
        super().__init__(api_key=api_key, **kwargs)
        self.default_model = model


# ============================================================================
# ANTHROPIC CLAUDE IMPLEMENTATION
# ============================================================================

class AnthropicAsyncOpenAI(openai.AsyncOpenAI):
    """
    AsyncOpenAI-compatible client that routes calls to Anthropic Claude API.
    
    This provides OpenAI API compatibility for direct Anthropic Claude models.
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str = None, **kwargs):
        """
        Initialize the Anthropic client.
        
        Args:
            model: Claude model to use (e.g., 'claude-sonnet-4-20250514')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            **kwargs: Additional args (absorbed to prevent errors)
        """
        super().__init__(api_key="anthropic-placeholder", **kwargs)
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Please set ANTHROPIC_API_KEY in your .env file or pass it explicitly"
            )
        
        self.anthropic_client = AsyncAnthropic(api_key=api_key)
        self.default_model = model
        self.chat = self.Chat(self)

    class Chat:
        """Chat interface that mimics openai.resources.chat.Chat"""
        
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            """Completions interface that mimics openai.resources.chat.Completions"""
            
            def __init__(self, parent):
                self.parent = parent
            
            async def create(
                self,
                model: str = None,
                messages: List[dict] = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
                response_model=None,
                **kwargs,
            ):
                """Create a chat completion."""
                model = model or self.parent.default_model
                
                # Extract system prompt
                system_prompt = ""
                claude_messages = []
                
                for msg in messages or []:
                    role = msg.get("role")
                    content = msg.get("content")
                    
                    if role == "system":
                        system_prompt = content
                    elif role in ["user", "assistant"]:
                        claude_messages.append({"role": role, "content": content})

                # Call Anthropic API
                response = await self.parent.anthropic_client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt if system_prompt else None,
                    messages=claude_messages
                )

                response_text = response.content[0].text

                # Create OpenAI-compatible response
                return self._create_response(response_text, model)
            
            def _create_response(self, text: str, model: str):
                """Create OpenAI-compatible response object."""
                class Message:
                    def __init__(self, content, role="assistant"):
                        self.content = content
                        self.role = role
                        self.function_call = None
                        self.tool_calls = None
                
                class Choice:
                    def __init__(self, message, index=0):
                        self.message = message
                        self.index = index
                        self.finish_reason = "stop"
                
                class Usage:
                    def __init__(self):
                        self.prompt_tokens = 0
                        self.completion_tokens = 0
                        self.total_tokens = 0
                
                class Response:
                    def __init__(self, text, model_used):
                        self.choices = [Choice(Message(text))]
                        self.model = model_used
                        self.id = "anthropic-completion"
                        self.object = "chat.completion"
                        self.created = 0
                        self.usage = Usage()
                    
                    def __getitem__(self, key):
                        return getattr(self, key)

                return Response(text, model)


# ============================================================================
# GOOGLE GEMINI IMPLEMENTATION
# ============================================================================

class GeminiAsyncOpenAI(openai.AsyncOpenAI):
    """
    AsyncOpenAI-compatible client that routes calls to Google Gemini API.
    
    This provides OpenAI API compatibility for Google Gemini models.
    """

    def __init__(self, model: str = "gemini-2.0-flash", api_key: str = None, **kwargs):
        """
        Initialize the Gemini client.
        
        Args:
            model: Gemini model to use (e.g., 'gemini-2.0-flash', 'gemini-1.5-pro')
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            **kwargs: Additional args (absorbed to prevent errors)
        """
        super().__init__(api_key="gemini-placeholder", **kwargs)
        
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Please set GOOGLE_API_KEY in your .env file or pass it explicitly"
            )
        
        self.gemini_client = genai.Client(api_key=api_key)
        self.default_model = model
        self.chat = self.Chat(self)

    class Chat:
        """Chat interface that mimics openai.resources.chat.Chat"""
        
        def __init__(self, parent):
            self.parent = parent
            self.completions = self.Completions(parent)

        class Completions:
            """Completions interface that mimics openai.resources.chat.Completions"""
            
            def __init__(self, parent):
                self.parent = parent
            
            async def create(
                self,
                model: str = None,
                messages: List[dict] = None,
                temperature: float = 0.0,
                max_tokens: int = 4096,
                response_model=None,
                **kwargs,
            ):
                """Create a chat completion."""
                model_name = model or self.parent.default_model
                
                # Convert OpenAI messages to Gemini format
                system_prompt = ""
                gemini_contents = []
                
                for msg in messages or []:
                    role = msg.get("role")
                    content = msg.get("content")
                    
                    if role == "system":
                        system_prompt = content
                    elif role == "user":
                        gemini_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
                    elif role == "assistant":
                        gemini_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

                # Build generation config
                generation_config = types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_prompt if system_prompt else None,
                )

                # Generate response using the new client API (async)
                response = await self.parent.gemini_client.aio.models.generate_content(
                    model=model_name,
                    contents=gemini_contents,
                    config=generation_config,
                )
                response_text = response.text

                # Create OpenAI-compatible response
                return self._create_response(response_text, model_name)
            
            def _create_response(self, text: str, model: str):
                """Create OpenAI-compatible response object."""
                class Message:
                    def __init__(self, content, role="assistant"):
                        self.content = content
                        self.role = role
                        self.function_call = None
                        self.tool_calls = None
                
                class Choice:
                    def __init__(self, message, index=0):
                        self.message = message
                        self.index = index
                        self.finish_reason = "stop"
                
                class Usage:
                    def __init__(self):
                        self.prompt_tokens = 0
                        self.completion_tokens = 0
                        self.total_tokens = 0
                
                class Response:
                    def __init__(self, text, model_used):
                        self.choices = [Choice(Message(text))]
                        self.model = model_used
                        self.id = "gemini-completion"
                        self.object = "chat.completion"
                        self.created = 0
                        self.usage = Usage()
                    
                    def __getitem__(self, key):
                        return getattr(self, key)

                return Response(text, model)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    
    async def test_bedrock():
        """Test the Bedrock client."""
        print("\n" + "="*60)
        print("Testing AWS Bedrock")
        print("="*60)
        
        try:
            llm = BedrockAsyncOpenAI()
            response = await llm.chat.completions.create(
                messages=[{"role": "user", "content": "Say hello in 5 words"}]
            )
            print(f"✓ Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def test_openai():
        """Test the OpenAI client."""
        print("\n" + "="*60)
        print("Testing OpenAI")
        print("="*60)
        
        try:
            llm = OpenAIAsyncClient(model="gpt-4o-mini")
            response = await llm.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say hello in 5 words"}]
            )
            print(f"✓ Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def test_anthropic():
        """Test the Anthropic client."""
        print("\n" + "="*60)
        print("Testing Anthropic Claude")
        print("="*60)
        
        try:
            llm = AnthropicAsyncOpenAI()
            response = await llm.chat.completions.create(
                messages=[{"role": "user", "content": "Say hello in 5 words"}]
            )
            print(f"✓ Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    async def test_gemini():
        """Test the Gemini client."""
        print("\n" + "="*60)
        print("Testing Google Gemini")
        print("="*60)
        
        try:
            llm = GeminiAsyncOpenAI()
            response = await llm.chat.completions.create(
                messages=[{"role": "user", "content": "Say hello in 5 words"}]
            )
            print(f"✓ Response: {response.choices[0].message.content}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Run all tests
    print("\n" + "="*60)
    print("MULTI-PROVIDER LLM CLIENT TESTS")
    print("="*60)
    
    asyncio.run(test_bedrock())
    asyncio.run(test_openai())
    asyncio.run(test_anthropic())
    asyncio.run(test_gemini())
    
    print("\n" + "="*60)
    print("All tests completed")
    print("="*60 + "\n")
