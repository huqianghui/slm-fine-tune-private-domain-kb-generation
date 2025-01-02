from abc import abstractmethod
from typing import List, Optional

from azure.ai.documentintelligence.models import (
    DocumentBarcode,
    DocumentFormula,
    DocumentParagraph,
    ParagraphRole,
)
from haystack.dataclasses import Document as HaystackDocument

from .elementInfo import ElementInfo
from .elementProcessor import DocumentElementProcessor
from .elementTools import replace_content_formulas_and_barcodes
from .sectionProcessor import get_heading_hashes
from .selectionMarkFormatter import SelectionMarkFormatter


class DocumentParagraphProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentParagraph elements.
    """

    expected_elements = [DocumentParagraph]

    @abstractmethod
    def convert_paragraph(
        self,
        element_info: ElementInfo,
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting paragraph content.
        """


class DefaultDocumentParagraphProcessor(DocumentParagraphProcessor):
    """
    The default processor for DocumentParagraph elements. This class provides a
    method for converting a DocumentParagraph element into a Haystack Document
    object that contains the text content. Since paragraphs can come with
    different roles, the text format string can be set for each possible role.

    The text format string can contain placeholders for the following:
    - {content}: The text content of the element
    - {heading_hashes}: The markdown heading hashes that set the heading
        level of the paragraph. The heading level is based on the section
        heirarchy of the paragraph and calculated automatically.

    :param general_text_format: Text format string for general text content.
    :type general_text_format: str, optional
    :param page_header_format: Text format string for page headers.
    :type page_header_format: str, optional
    :param page_footer_format: Text format string for page footers.
    :type page_footer_format: str, optional
    :param title_format: Text format string for titles.
    :type title_format: str, optional
    :param section_heading_format: Text format string for section headings.
    :type section_heading_format: str, optional
    :param footnote_format: Text format string for footnotes.
    :type footnote_format: str, optional
    :param formula_format: Text format string for formulas.
    :type formula_format: str, optional
    :param page_number_format: Text format string for page numbers.
    :type page_number_format: str, optional
    """

    expected_format_placeholders = ["{content}", "{heading_hashes}"]

    def __init__(
        self,
        general_text_format: Optional[str] = "{content}",
        page_header_format: Optional[str] = None,
        page_footer_format: Optional[str] = None,
        title_format: Optional[str] = "\n{heading_hashes} **{content}**",
        section_heading_format: Optional[str] = "\n{heading_hashes} **{content}**",
        footnote_format: Optional[str] = "*Footnote:* {content}",
        formula_format: Optional[str] = "*Formula:* {content}",
        page_number_format: Optional[str] = None,
    ):
        self.paragraph_format_mapper = {
            None: general_text_format,
            ParagraphRole.PAGE_HEADER: page_header_format,
            ParagraphRole.PAGE_FOOTER: page_footer_format,
            ParagraphRole.TITLE: title_format,
            ParagraphRole.SECTION_HEADING: section_heading_format,
            ParagraphRole.FOOTNOTE: footnote_format,
            ParagraphRole.FORMULA_BLOCK: formula_format,
            ParagraphRole.PAGE_NUMBER: page_number_format,
        }

    def convert_paragraph(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a paragraph element into a Haystack Document object containing
        the text content of the paragraph.

        :param element_info: Element information for the paragraph.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the line.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the paragraph content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        format_mapper = self.paragraph_format_mapper[element_info.element.role]
        heading_hashes = get_heading_hashes(
            element_info.section_heirarchy_incremental_id
        )
        content = replace_content_formulas_and_barcodes(
            element_info.element.content,
            element_info.element.spans,
            all_formulas,
            all_barcodes,
        )
        content = selection_mark_formatter.format_content(content)
        if format_mapper:
            formatted_text = format_mapper.format(
                heading_hashes=heading_hashes, content=content
            )
            formatted_if_null = format_mapper.format(heading_hashes="", content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=formatted_text,
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()