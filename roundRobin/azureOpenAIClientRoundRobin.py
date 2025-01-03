import asyncio
import json
import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AzureOpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv(verbose=True)

azureOpenAIRoundRobinConnection= os.getenv("AZURE_OPENAI_ROUND_ROBIN_CONNETION","[]")
apiVersion = os.getenv("AZURE_OPENAI_ROUND_ROBIN_API_VERSION","2024-10-21") # this might change in the future
deploymentName = os.getenv("AZURE_OPENAI_ROUND_ROBIN_DEPLOYMENT_NAME","gpt-4o")

@dataclass
class AzureOpenAIConnection:
    endpoint: str
    apiKey: str

class AzureOpenAIClientsRoundRobin:
    def __init__(self):
        self.clients = _build_azure_oppen_AI_async_clients(azureOpenAIRoundRobinConnection)
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

def _load_connections(azureOpenAIRoundRobinConnection:str)->list[AzureOpenAIConnection]:
    data_list = json.loads(azureOpenAIRoundRobinConnection)
    return [
        AzureOpenAIConnection(
            endpoint=item["AZURE_OPENAI_ENDPOINT"],
            apiKey=item["AZURE_OPENAI_API_KEY"],
        )
        for item in data_list
    ]
    
def _build_azure_oppen_AI_async_clients(azureOpenAIRoundRobinConnection:str)->list[AsyncAzureOpenAI]:
    azureOpenAIConnectionList = _load_connections(azureOpenAIRoundRobinConnection)
    clients = []
    for connection in azureOpenAIConnectionList:
        client = AsyncAzureOpenAI(
            azure_endpoint=connection.endpoint,
            azure_deployment=deploymentName,
            api_key=connection.apiKey,
            api_version=apiVersion)
        clients.append(client)
    return clients

# init client
client_manager = AzureOpenAIClientsRoundRobin()

