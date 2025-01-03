import asyncio
import os

from roundRobin.azureOpenAIClientRoundRobin import client_manager


async def roundRonbinTest():
    # step1) Get Doc Intelligence result
    for i in range(10):
        aAzureOpenclient = await client_manager.get_next_client()
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