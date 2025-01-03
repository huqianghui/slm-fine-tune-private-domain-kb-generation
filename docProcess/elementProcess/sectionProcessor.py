from abc import abstractmethod
from typing import List, Optional

from azure.ai.documentintelligence.models import DocumentSection
from haystack.dataclasses import Document as HaystackDocument

from .elementInfo import ElementInfo
from .elementProcessor import DocumentElementProcessor


async def get_heading_hashes(section_heirarchy: Optional[tuple[int]]) -> str:
    """
    Gets the heading hashes for a section heirarchy.

    :param section_heirarchy: Tuple of section IDs representing the section
        heirarchy - For example, (1,1,4).
    :type section_heirarchy: tuple[int], optional
    :return: A string containing the heading hashes.
    :rtype: str
    """
    if section_heirarchy:
        return "#" * len(section_heirarchy)
    return ""

class DocumentSectionProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentSection elements.
    """

    expected_elements = [DocumentSection]

    @abstractmethod
    async def convert_section(
        self,
        element_info: ElementInfo,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for processing section elements.
        """

class DefaultDocumentSectionProcessor(DocumentSectionProcessor):
    """
    The default processor for DocumentSection elements. This class provides
    methods for converting a DocumentSection element into Haystack Documents
    that contain section information.

    The text_format string can contain placeholders for the following:
    - {section_incremental_id}: The heirarchical ID of the section
        (e.g. (1.2.4)).

    :param text_format: A text format string which defines how the content
        should be extracted from the element. If set to None, no text content
        will be extracted from the element.
    :type text_format: str, optional
    :param max_heirarchy_depth: The maximum depth of the heirarchy to include in
        the section incremental ID. If a section has an ID that is deeper than
        max_heirarchy_depth, it's section ID will be ignored. If None, all
        section ID levels will be included when printing section IDs.
    :type max_heirarchy_depth: int, optional
    """

    def __init__(
        self,
        text_format: Optional[str] = None,
        max_heirarchy_depth: Optional[int] = 3,
    ):
        self.text_format = text_format
        # If max_depth is None, ensure it is set to a high number
        if max_heirarchy_depth < 0:
            raise ValueError("max_depth must be a positive integer or None.")
        self.max_heirarchy_depth = (
            max_heirarchy_depth if max_heirarchy_depth is not None else 999
        )

    async def convert_section(
        self,
        element_info: ElementInfo,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a DocumentSection element into a Haystack Document.

        :param element_info: Information about the DocumentSection element.
        :type element_info: ElementInfo
        :param section_heirarchy: The heirarchical ID of the section.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the section information.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        # Section incremental ID can be empty for the first section representing
        # the entire document. Only output text if values exist, and only if max_depth
        # is not exceeded. We want to avoid printing the same section ID -
        # e.g. (3, 1) should become 3.1 but (3, 1, 1) should be ignored if
        # max_depth is only 2.
        if self.text_format and element_info.section_heirarchy_incremental_id:
            if (
                len(element_info.section_heirarchy_incremental_id)
                <= self.max_heirarchy_depth
            ):
                section_incremental_id_text = ".".join(
                    str(i)
                    for i in element_info.section_heirarchy_incremental_id[
                        : self.max_heirarchy_depth
                    ]
                )
            else:
                section_incremental_id_text = ""
            formatted_text = self.text_format.format(
                section_incremental_id=section_incremental_id_text
            )
            formatted_if_null = self.text_format.format(section_incremental_id="")
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