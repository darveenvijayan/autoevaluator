import os
from openai import OpenAI, AzureOpenAI
import instructor

def setup_client():
    if os.environ["DEPLOYMENT"] == "azure":
        client = AzureOpenAI(
        api_version="2023-07-01-preview"
        )
        model = "gpt-35-turbo"
    else:
        client = OpenAI()
        model = "gpt-3.5-turbo"

    # Apply the patch to the OpenAI client
    client = instructor.from_openai(client)
    return client, model