# coding: utf-8

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

"""
FILE: azureDocIntellij.py

DESCRIPTION:
    This code process the pdf file to extract all elements including figures from the document to analyze.

    This code uses Layout model to process.

USAGE:
    python azureDocIntellij.py

    Set the environment variables with your own values before running the sample:
    1) DOCUMENTINTELLIGENCE_ENDPOINT - the endpoint to your Document Intelligence resource.
    2) DOCUMENTINTELLIGENCE_API_KEY - your Document Intelligence API key.
"""

import logging
import os

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

from .fileTools import convert_pdf_to_base64

load_dotenv(override=True)
logger = logging.getLogger(__name__)


DOC_INTEL_MODEL_ID = "prebuilt-layout"  # Set Document Intelligence model ID
DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
DOC_INTEL_API_KEY = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
DOC_INTEL_FEATURES = ['ocrHighResolution', 'languages', 'styleFont']


# Set up the Document Intelligence v4.0 preview client. This will allow us to
# use the latest features of the Document Intelligence service including figure extraction.
di_client = DocumentIntelligenceClient(
    endpoint=DOC_INTEL_ENDPOINT,
    credential=AzureKeyCredential(DOC_INTEL_API_KEY),
    api_version="2024-07-31-preview",
)


def get_analyze_document_result(
    pdf_file_pth: str,
    model_id: str = "prebuilt-layout",
    **kwargs
) -> AnalyzeResult:
    """
    Gets the AnalyzeResult for the PDF file using a Document Intelligence
    client.
    """
    analyze_request = AnalyzeDocumentRequest(bytes_source=convert_pdf_to_base64(pdf_file_pth))
    poller = di_client.begin_analyze_document(
            model_id=model_id,
            analyze_request=analyze_request,
            features=DOC_INTEL_FEATURES,
            **kwargs)
    
    # original result from the document intellijence
    analyzeResult = poller.result()
    return analyzeResult


