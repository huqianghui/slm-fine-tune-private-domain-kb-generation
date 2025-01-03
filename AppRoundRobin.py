import asyncio
import os

from roundRobin.azureOpenAIClientRoundRobin import AzureOpenAIClientsRoundRobin


async def roundRonbinTest():
    # step1) Get Doc Intelligence result
    azureOpenAIClientsRoundRobin = AzureOpenAIClientsRoundRobin()
    for i in range(10):
        aAzureOpenclient = await azureOpenAIClientsRoundRobin.get_next_client()
        response = await aAzureOpenclient.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_ROUND_ROBIN_DEPLOYMENT_NAME","gpt-4o"),
            messages=[
            {
                "role": "user",
                "content": "hello!"
            }]
        )
        print(f"resposne from {i} aAzureOpenclient " + response.choices[0].message.content) 
    
 
if __name__ == "__main__":
  reulst = asyncio.run(roundRonbinTest())