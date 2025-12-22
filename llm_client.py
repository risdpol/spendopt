from openai import OpenAI
 
endpoint = "https://new-ai-project.openai.azure.com/openai/v1/"
deployment_name = "gpt-4.1-mini"
api_key = "7UgtmPvCZ6Yyj59aTXlQDKQz7YrIncDQgV3TNmoUdn4HTSlKgDMKJQQJ99BIACYeBjFXJ3w3AAAAACOGLmgg"

def create_client():
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key
    )
    return client

# client = create_client()
# completion = client.chat.completions.create(
#     model=deployment_name,
#     messages=[
#         {
#             "role": "user",
#             "content": "What is the capital of France?",
#         }
#     ],
#     temperature=0.7,
# )

# print(completion.choices[0].message) #print the response message