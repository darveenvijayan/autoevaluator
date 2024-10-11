import os
from openai import OpenAI, AzureOpenAI
import instructor


def setup_client(model_name='gpt-4o-mini'):
    try:
        deployment = os.environ["DEPLOYMENT"]
    except KeyError:
        raise EnvironmentError("The environment variable 'DEPLOYMENT' is not set. Please set it to 'azure' or 'not-azure'. Example: ``` export DEPLOYMENT='non-azure' ```")

    if deployment == "azure":
        try:
            client = AzureOpenAI(api_version="2023-07-01-preview")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AzureOpenAI client: {e}")
        model = model_name
    else:
        try:
            client = OpenAI()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
        model = model_name

    try:
        client = instructor.from_openai(client)
    except Exception as e:
        raise RuntimeError(f"Failed to apply the patch to the OpenAI client: {e}")

    return client, model


# def setup_client(deployment=None, api_version="2023-07-01-preview", model_azure="gpt-35-turbo", model_openai="gpt-4o-mini"):
#     if deployment is None:
#         try:
#             deployment = os.environ["DEPLOYMENT"]
#         except KeyError:
#             raise EnvironmentError("The environment variable 'DEPLOYMENT' is not set. Please set it to 'azure' or 'not-azure'. Example: ``` export DEPLOYMENT='non-azure' ```")

#     if deployment == "azure":
#         try:
#             client = AzureOpenAI(api_version=api_version)
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize AzureOpenAI client: {e}")
#         model = model_azure
#     else:
#         try:
#             client = OpenAI()
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
#         model = model_openai

#     try:
#         client = instructor.from_openai(client)
#     except Exception as e:
#         raise RuntimeError(f"Failed to apply the patch to the OpenAI client: {e}")

#     return client, model

