import asyncio
import logging
import os

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(verbose=True)

azureOpenAIRoundRobinConnection= os.getenv("AZURE_OPENAI_ROUND_ROBIN_CONNETION","")
apiVersion = os.getenv("AZURE_OPENAI_ROUND_ROBIN_API_VERSION","2024-10-01-preview") # this might change in the future
deploymentName = os.getenv("AZURE_OPENAI_ROUND_ROBIN_DEPLOYMENT_NAME","gpt-4o")

class AzureOpenAIClientsRoundRobin:
    def __init__(self):
        self.clients = build_azure_oppen_AI_async_clients(azureOpenAIRoundRobinConnection)
        self.client_count = len(self.clients)
        self.index = 0  # init
        self.lock = asyncio.Lock()
    
    async def get_next_client(self):
        async with self.lock: 
            # get client
            client = self.clients[self.index]
            # update index 
            self.index = (self.index + 1) % self.client_count
            return client
    
def build_azure_oppen_AI_async_clients(azureOpenAIRoundRobinConnection:str)->list[AsyncAzureOpenAI]:
        pass

# init client
client_manager = AzureOpenAIClientsRoundRobin()

