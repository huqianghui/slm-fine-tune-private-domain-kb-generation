import asyncio
import os

from docProcess.azureDocIntellig import get_analyze_document_result
from docProcess.azureDocIntelligResultPostProcessor import (
    DocumentIntelligenceResultPostProcessor,
)
from docProcess.customizedProcess.imgeDescriptionByLlmFigureProcessor import (
    ImgeDescriptionByLlmFigureProcessor,
)
from docProcess.customizedProcess.markdownImageTagFigureProcessor import (
    MarkdownImageTagDocumentFigureProcessor,
)
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
from docProcess.fileTools import extract_pdf_page_images, load_pymupdf_pdf

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
    before_table_text_formats=["*Table Caption:* {caption}"],
    after_table_text_formats=None,
)

# add content to text format
figure_processor = DefaultDocumentFigureProcessor(
    before_figure_text_formats=["*Figure Caption:* {caption}"],
    output_figure_img=False,
    figure_img_text_format="*Figure Content:*\n{content}",
    after_figure_text_formats=None,
)

markdown_img_tag_path_or_url = os.getenv("MARKDOWN_IMG_TAG_PATH_OR_URL")

# markdown_figure_processor = MarkdownImageTagDocumentFigureProcessor(
#     before_figure_text_formats=["*Figure Caption:* {caption}"],
#     output_figure_img=True,
#     figure_img_text_format="*Figure Content:*\n{content}",
#     after_figure_text_formats=None,
#     markdown_img_tag_path_or_url=markdown_img_tag_path_or_url
# )

imgeDescriptionByLlmFigureProcessor = ImgeDescriptionByLlmFigureProcessor(
   before_figure_text_formats=["*Figure Caption:* {caption}"],
   output_figure_img=True,
   figure_img_text_format="*Figure Content:*\n{content}",
   after_figure_text_formats=None
)

markdown_figure_processor = MarkdownImageTagDocumentFigureProcessor(
    before_figure_text_formats=["*Figure Caption:* {caption}"],
    output_figure_img=True,
    figure_img_text_format="*Figure Content:*\n{content}",
    after_figure_text_formats=None,
    markdown_img_tag_path_or_url=markdown_img_tag_path_or_url
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
doc_intel_result_processor = DocumentIntelligenceResultPostProcessor(
    page_processor = page_processor,
    section_processor = section_processor,
    table_processor = table_processor,
    figure_processor = imgeDescriptionByLlmFigureProcessor,
    paragraph_processor = paragraph_processor,
    line_processor = line_processor,
    word_processor = word_processor,
    selection_mark_formatter = selection_mark_formatter
)

async def processPDF2Markdown(pdf_path:str)->str:
    # step1) Get Doc Intelligence result
    di_original_result = await get_analyze_document_result(pdf_path)

    # step2) convert pdf page to images
    pdf = load_pymupdf_pdf(pdf_path=pdf_path, pdf_url=None)
    doc_page_imgs = extract_pdf_page_images(pdf, img_dpi=100, starting_idx=1)

    # step2) Process result with the element processor
    processed_content_docs = await doc_intel_result_processor.process_analyze_result(
        di_original_result, 
        doc_page_imgs=doc_page_imgs,
        on_error="raise")
    

    filtered_docs = [doc for doc in processed_content_docs if  doc.meta["element_type"] != "DocumentPage"]
    markdownResult = await convert_processed_di_docs_to_markdown(filtered_docs, default_text_merge_separator="\n")
    print(markdownResult)
    return markdownResult
 
if __name__ == "__main__":
  reulst = asyncio.run(processPDF2Markdown("raw_documents/pdf/oral_cancer_text_5th_table&image.pdf"))