import asyncio
import os
import shutil

import tiktoken
from dotenv import load_dotenv
from langchain.text_splitter import MarkdownHeaderTextSplitter
from tenacity import retry, stop_after_attempt, wait_random_exponential

from cache.cacheConfig import async_diskcache, cache

from .dataMode import MergedChunk, MergedChunkFile, SplitResult
from .semanticChunk import process_big_chunk_file, process_small_chunk_file

load_dotenv()


headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3")
]

text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,return_each_line=False,strip_headers=False)

# set the path to save the merged chunk file
SPLIT_CHUNK_FILE_PATH = os.getenv("SPLIT_CHUNK_FILE_PATH","processed_documents/markdown/splitChunk/")
if not os.path.exists(SPLIT_CHUNK_FILE_PATH):
    os.makedirs(SPLIT_CHUNK_FILE_PATH)

#@async_diskcache("split_content_by_markdown_header")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def splitContentByMarkdownHeader(docMarkdownStr:str,filename:str)->list[SplitResult]:
    splits = text_splitter.split_text(docMarkdownStr)
    encoding = tiktoken.get_encoding(os.getenv("LLM_CODER","o200k_base"))
    splitResult = []
    for idx, split in enumerate(splits): 
        split_token_count = len(encoding.encode(split.page_content))
        splitResult.append(SplitResult(tokens=split_token_count,content=split.page_content))
        with open(SPLIT_CHUNK_FILE_PATH + filename + f"_split_{idx}.md", "w", encoding="utf-8") as f:
            f.write(split.page_content)
    return splitResult


# set the 
SPIPPETS_SIZE=int(os.getenv("SPIPPETS_SIZE","600"))
# set the miminum size of the split content  
CHUNK_MIN_SIZE = int(os.getenv("CHUNK_MIN_SIZE","1000"))  
# set the maximum size of the split content
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE","1400"))
# set the abusolute maximum size of the split content
CHUNK_ABUSOLUTE_MAX_SIZE = int(os.getenv("CHUNK_ABUSOLUTE_MAX_SIZE","2400"))



async def mergeSpitsIntoChunk(splitResult: list[SplitResult]) -> list[MergedChunk]:
    # store the merged chunks  
    mergedChunkList = []  
    index = 0  # current index of the split result
    while index < len(splitResult):
        print("...processing the split: " + str(index))  
        currentSplit = splitResult[index]
        currentSplitContent = currentSplit.content
        currentSplitTokens = currentSplit.tokens

        # if the current split is bigger than the maximum size, keep it as is
        if CHUNK_MIN_SIZE <= currentSplitTokens:  
            mergedChunk = MergedChunk(splits=currentSplitContent, totalTokens=currentSplitTokens, note="it is bigger than chunk min size,keep as is")
            mergedChunkList.append(mergedChunk)  
            index += 1   
        # if the current split is smaller than the minimum size, merge it with the next split
        else:
            combinedTokens = currentSplitTokens  
            combinedContent = currentSplitContent
            index += 1  
    
            while combinedTokens < CHUNK_MAX_SIZE and index < len(splitResult):
                currentSplit = splitResult[index]
                currentSplitContent = currentSplit.content
                currentSplitTokens = currentSplit.tokens 
                combinedTokens += currentSplitTokens

                if combinedTokens < CHUNK_MAX_SIZE or currentSplitTokens < SPIPPETS_SIZE or ((combinedTokens -currentSplitTokens) < SPIPPETS_SIZE) :
                    # if combined tokens is less than the maximum size, add the current split to the combined content or the chunk is less than the spippet size
                    combinedContent += "\n" + currentSplitContent
                    index += 1
                else:
                    # remove the last split if the combined tokens is greater than the maximum size
                    combinedTokens -=currentSplitTokens
                    break  
                
            mergedChunk = MergedChunk(splits=combinedContent, totalTokens=combinedTokens, note="it is bigger than chunk min size,keep as is")
            mergedChunkList.append(mergedChunk)
        
    # if the final split is smaller than the minimum size, merge it with the previous split
    lastMergedChunk = mergedChunkList.pop()
    if lastMergedChunk.totalTokens < CHUNK_MIN_SIZE:
        secondLastMergedChunk = mergedChunkList.pop()
        secondLastMergedChunk.splits += lastMergedChunk.splits
        secondLastMergedChunk.totalTokens += lastMergedChunk.totalTokens
        mergedChunkList.append(secondLastMergedChunk)

    return mergedChunkList

# set the path to save the merged chunk file
MERGE_CHUNK_FILE_PATH = os.getenv("MERGE_CHUNK_FILE_PATH","processed_documents/markdown/mergedChunk/")
if not os.path.exists(MERGE_CHUNK_FILE_PATH):
    os.makedirs(MERGE_CHUNK_FILE_PATH)

async def saveMergedChunkIntoFile(mergedChunkList: list[MergedChunk],filename:str)->list[MergedChunkFile]:

    MergedChunkFileList = []
    for idx, chunk in enumerate(mergedChunkList):  
        file_name = f"{filename}_{idx}_chunk_tokens_{chunk.totalTokens}.md"
        abPath = MERGE_CHUNK_FILE_PATH + file_name
        
        mergedChunkFile=MergedChunkFile(filePath=abPath,totalTokens=chunk.totalTokens)
        MergedChunkFileList.append(mergedChunkFile)
        
        with open(abPath, "w", encoding="utf-8") as file:
            file.write(chunk.splits)
    return MergedChunkFileList

async def processMergdeChunkFile(mergedChunkFileList:list[MergedChunkFile]):
    chunkTasks = [process_big_chunk_file(mergedChunkFile.filePath) for mergedChunkFile in mergedChunkFileList if mergedChunkFile.totalTokens >= CHUNK_ABUSOLUTE_MAX_SIZE]
    copyTasks = [process_small_chunk_file(mergedChunkFile.filePath) for mergedChunkFile in mergedChunkFileList if mergedChunkFile.totalTokens < CHUNK_ABUSOLUTE_MAX_SIZE]
    await asyncio.gather(*copyTasks)
    # run the tasks concurrently
    results = await asyncio.gather(*chunkTasks)
    print(results)





