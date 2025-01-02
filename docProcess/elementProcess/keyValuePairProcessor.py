from abc import abstractmethod
from typing import List, Optional

from azure.ai.documentintelligence.models import (
    DocumentBarcode,
    DocumentFormula,
    DocumentKeyValuePair,
)
from haystack.dataclasses import Document as HaystackDocument

from .elementInfo import ElementInfo
from .elementProcessor import DocumentElementProcessor
from .elementTools import replace_content_formulas_and_barcodes
from .selectionMarkFormatter import SelectionMarkFormatter


class DocumentKeyValuePairProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentKeyValuePair elements.
    """

    expected_elements = [DocumentKeyValuePair]

    @abstractmethod
    def convert_kv_pair(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting Key Value pairs.
        """


class DefaultDocumentKeyValuePairProcessor(DocumentKeyValuePairProcessor):
    """
    The default processor for DocumentKeyValuePair elements. This class provides
    a method for converting a DocumentKeyValuePair element into a
    Haystack Document object that contains the text content.

    The text format string can contain placeholders for the following:
    - {key_content}: The text content of the key.
    - {value_content}: The text content of the value.

    :param text_format: Text format string for text content.
    :type text_format: str, optional
    """

    expected_format_placeholders = ["{key_content}", "{value_content}"]

    def __init__(
        self,
        text_format: Optional[str] = "*Key Value Pair*: {key_content}: {value_content}",
    ):
        self.text_format = text_format

    def convert_kv_pair(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a DocumentKeyValuePair element into a Haystack Document object
        containing the text contents of the element.

        :param element_info: Element information for the KV pair.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the word.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the KV pair content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        key_content = replace_content_formulas_and_barcodes(
            element_info.element.key.content,
            element_info.element.key.spans,
            all_formulas,
            all_barcodes,
        )
        key_content = selection_mark_formatter.format_content(key_content)
        value_content = replace_content_formulas_and_barcodes(
            element_info.element.value.content,
            element_info.element.value.spans,
            all_formulas,
            all_barcodes,
        )
        value_content = selection_mark_formatter.format_content(value_content)
        if self.text_format:
            formatted_text = self.text_format.format(
                key_content=key_content, value_content=value_content
            )
            formatted_if_null = self.text_format.format(
                key_content="", value_content=""
            )
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