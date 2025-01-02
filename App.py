import asyncio
import json
import logging
import os

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from promptflow.tracing import start_trace, trace
from tenacity import retry, stop_after_attempt, wait_random_exponential

from docProcess.azureDocIntellig import get_analyze_document_result
from docProcess.azureDocIntelligResultPostProcessor import DocumentIntelligenceProcessor
from docProcess.docIntelligElementTools import convert_processed_di_docs_to_markdown
from docProcess.elementProcess.figureProcessor import DefaultDocumentFigureProcessor
from docProcess.elementProcess.keyValuePairProcessor import (
    DefaultDocumentKeyValuePairProcessor,
)
from docProcess.elementProcess.lineProcessor import DefaultDocumentLineProcessor
from docProcess.elementProcess.pageProcessor import DefaultDocumentPageProcessor
from docProcess.elementProcess.paragraphProcessor import (
    DefaultDocumentParagraphProcessor,
)
from docProcess.elementProcess.sectionProcessor import DefaultDocumentSectionProcessor
from docProcess.elementProcess.selectionMarkFormatter import (
    DefaultSelectionMarkFormatter,
)
from docProcess.elementProcess.tableProcessor import DefaultDocumentTableProcessor
from docProcess.elementProcess.wordProcessor import DefaultDocumentWordProcessor

selection_mark_formatter = DefaultSelectionMarkFormatter(
    selected_replacement="[X]", 
    unselected_replacement="[ ]"
)
section_processor = DefaultDocumentSectionProcessor(
    text_format=None,
)
page_processor = DefaultDocumentPageProcessor(
    page_start_text_formats=["\n*Page {page_number} content:*"],
    page_end_text_formats=None,
    page_img_order="after",
    page_img_text_intro="*Page {page_number} Image:*",
    img_export_dpi=100,
    adjust_rotation = True,
    rotated_fill_color = (255, 255, 255),
)
table_processor = DefaultDocumentTableProcessor(
    before_table_text_formats=["**Table {table_number} Info**\n", "*Table Caption:* {caption}", "*Table Footnotes:* {footnotes}", "*Table Content:*"],
    after_table_text_formats=None,
)

# add content to text format
figure_processor = DefaultDocumentFigureProcessor(
    before_figure_text_formats=["**Figure {figure_number} Info**\n", "*Figure Caption:* {caption}", "*Figure Footnotes:* {footnotes}", "*Figure Content:*\n{content}"],
    output_figure_img=True,
    figure_img_text_format="*Figure Content:*\n{content}",
    after_figure_text_formats=None,
)
key_value_pair_processor = DefaultDocumentKeyValuePairProcessor(
    text_format = "*Key Value Pair*: {key_content}: {value_content}",
)
paragraph_processor = DefaultDocumentParagraphProcessor(
    general_text_format = "{content}",
    page_header_format = None,
    page_footer_format = None,
    title_format = "\n{heading_hashes} **{content}**",
    section_heading_format = "\n{heading_hashes} **{content}**",
    footnote_format = "*Footnote:* {content}",
    formula_format = "*Formula:* {content}",
    page_number_format = None,
)
line_processor = DefaultDocumentLineProcessor()
word_processor = DefaultDocumentWordProcessor()

# Now construct the DocumentIntelligenceProcessor class which uses each of these sub-processors
doc_intel_result_processor = DocumentIntelligenceProcessor(
    page_processor = page_processor,
    section_processor = section_processor,
    table_processor = table_processor,
    figure_processor = figure_processor,
    paragraph_processor = paragraph_processor,
    line_processor = line_processor,
    word_processor = word_processor,
    selection_mark_formatter = selection_mark_formatter
)

async def processPDF(pdf_path:str):
    # Get Doc Intelligence result
    di_original_result = get_analyze_document_result(pdf_path)

    # Process the API response with the processor
    processed_content_docs = doc_intel_result_processor.process_analyze_result(
        di_original_result, 
        on_error="ignore")

    filtered_docs = [doc for doc in processed_content_docs if  doc.meta["element_type"] != "DocumentPage"]
    print(convert_processed_di_docs_to_markdown(filtered_docs, default_text_merge_separator="\n"))


 
if __name__ == "__main__":
  reulst = asyncio.run(processPDF("raw_documents/pdf/oral_cancer_text_5th_table&image.pdf"))