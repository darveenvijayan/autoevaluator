import os
from openai import OpenAI, AzureOpenAI
import instructor


def setup_client():
    try:
        deployment = os.environ["DEPLOYMENT"]
    except KeyError:
        raise EnvironmentError("The environment variable 'DEPLOYMENT' is not set. Please set it to 'azure' or 'not-azure'. Example: ``` export DEPLOYMENT='non-azure' ```")

    if deployment == "azure":
        try:
            client = AzureOpenAI(api_version="2023-07-01-preview")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AzureOpenAI client: {e}")
        model = "gpt-35-turbo"
    else:
        try:
            client = OpenAI()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        model = "gpt-3.5-turbo"

    try:
        client = instructor.from_openai(client)
    except Exception as e:
        raise RuntimeError(f"Failed to apply the patch to the OpenAI client: {e}")

    return client, model
