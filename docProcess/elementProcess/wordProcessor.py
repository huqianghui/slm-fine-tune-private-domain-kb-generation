from abc import abstractmethod
from typing import List, Optional

from azure.ai.documentintelligence.models import (
    DocumentBarcode,
    DocumentFormula,
    DocumentWord,
)
from haystack.dataclasses import Document as HaystackDocument

from .elementInfo import ElementInfo
from .elementProcessor import DocumentElementProcessor
from .elementTools import replace_content_formulas_and_barcodes
from .selectionMarkFormatter import SelectionMarkFormatter


class DocumentWordProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentWord elements.
    """

    expected_elements = [DocumentWord]

    @abstractmethod
    def convert_word(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting word content.
        """

class DefaultDocumentWordProcessor(DocumentWordProcessor):
    """
    The default processor for DocumentWord elements. This class provides a
    method for converting a DocumentWord element into a Haystack Document object
    that contains the text content.

    The text format string can contain placeholders for the following:
    - {content}: The text content of the element

    :param text_format: A text format string which defines
        the content which should be extracted from the element. If set to None,
        no text content will be extracted from the element.
    :type text_format: str, optional
    """

    def __init__(
        self,
        text_format: Optional[str] = "{content}",
    ):
        self.text_format = text_format

    def convert_word(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a word element into a Haystack Document object containing the
        text content of the word.

        :param element_info: Element information for the word.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the word.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the word content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        content = replace_content_formulas_and_barcodes(
            element_info.element.content,
            [element_info.element.span],
            all_formulas,
            all_barcodes,
        )
        content = selection_mark_formatter.format_content(content)
        if self.text_format:
            formatted_text = self.text_format.format(content=content)
            formatted_if_null = self.text_format.format(content="")
            if formatted_text != formatted_if_null:
                return [
                    HaystackDocument(
                        id=f"{element_info.element_id}",
                        content=self.text_format.format(content=content),
                        meta={
                            "element_id": element_info.element_id,
                            "element_type": type(element_info.element).__name__,
                            "page_number": element_info.start_page_number,
                            "section_heirarchy": section_heirarchy,
                        },
                    )
                ]
        return list()

