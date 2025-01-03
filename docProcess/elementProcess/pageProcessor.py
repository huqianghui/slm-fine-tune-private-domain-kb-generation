from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

from azure.ai.documentintelligence.models import AnalyzeResult, Document, DocumentPage
from haystack.dataclasses import ByteStream as HaystackByteStream
from haystack.dataclasses import Document as HaystackDocument
from PIL.Image import Image as PILImage

from ..imageTools import TransformedImage, pil_img_to_base64, rotate_img_pil
from .elementInfo import ElementInfo, SpanBounds
from .elementProcessor import DocumentElementProcessor
from .elementTools import get_min_and_max_span_bounds


class DocumentPageProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentPage elements.
    """

    expected_elements = [DocumentPage]

    @abstractmethod
    async def export_page_img(
        self, pdf_page_img: PILImage, di_page: DocumentPage
    ) -> TransformedImage:
        """
        Method for exporting the page image and applying transformations.
        """

    @abstractmethod
    async def convert_page_start(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting content for the beginning of the page.
        """

    @abstractmethod
    async def convert_page_end(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting content for the end of the page.
        """


class DefaultDocumentPageProcessor(DocumentPageProcessor):
    """
    The default processor for DocumentPage elements. This class provides
    methods for converting a DocumentPage element into Haystack Documents
    that contain the text and page image content.

    The text format string can contain placeholders for the following:
    - {page_number}: The 1-based page number of the page.

    :param page_start_text_formats: Text formats to be used to export text
        content that should be shown at the start of the page. If None, no text
        content will be exported at the start of the page.
    :type page_start_text_formats: List[str], optional
    :param page_end_text_formats: Text formats to be used to export text
        content that should be shown at the end of the page. If None, no text
        content will be exported at the end of the page.
    :type page_end_text_formats: str, optional
    :param page_img_order: Whether to export the page image 'before' or 'after'
        the rest of the page content, or not at all ('None').
    :type page_img_order: Literal["before", "after"], optional
    :param page_img_text_intro: Introduction text to be paired with the page
        image, defaults to "*Page Image:*"
    :type page_img_text_intro: str, optional
    :param img_export_dpi: DPI of the exported page image, defaults to 100
    :type img_export_dpi: int, optional
    :param adjust_rotation: Whether to automatically adjust the image based on
        the page angle detected by Document Intelligence, defaults to True
    :type adjust_rotation: bool, optional
    :param rotated_fill_color: Defines the fill color to be used if the page
        image is rotated and new pixels are added to the image, defaults to
        (255, 255, 255)
    :type rotated_fill_color: Tuple[int], optional
    """

    expected_format_placeholders = ["{page_number}"]

    def __init__(
        self,
        page_start_text_formats: Optional[List[str]] = [
            "\n*Page {page_number} content:*\n"
        ],
        page_end_text_formats: Optional[str] = None,
        page_img_order: Optional[Literal["before", "after"]] = "after",
        page_img_text_intro: Optional[str] = "*Page {page_number} Image:*",
        img_export_dpi: int = 100,
        adjust_rotation: bool = True,
        rotated_fill_color: Tuple[int] = (255, 255, 255),
    ):
        self.page_start_text_formats = page_start_text_formats
        self.page_end_text_formats = page_end_text_formats
        self.page_img_order = page_img_order
        self.page_img_text_intro = page_img_text_intro
        self.img_export_dpi = img_export_dpi
        self.adjust_rotation = adjust_rotation
        self.rotated_fill_color = rotated_fill_color

    async def export_page_img(
        self, pdf_page_img: PILImage, di_page: DocumentPage
    ) -> TransformedImage:
        """
        Export the page image and apply transformations.

        :param pdf_page_img: PIL Image of the page.
        :type pdf_page_img: PIL.Image.Image
        :param di_page: DocumentPage object.
        :type di_page: DocumentPage
        :return: Transformed image object containing the original and
            transformed image information.
        :rtype: TransformedImage
        """
        if self.adjust_rotation:
            transformed_img = rotate_img_pil(
                pdf_page_img, di_page.angle or 0, self.rotated_fill_color
            )
        else:
            transformed_img = pdf_page_img
        return TransformedImage(
            image=transformed_img,
            orig_image=pdf_page_img,
            rotation_applied=di_page.angle if self.adjust_rotation else None,
        )

    async def convert_page_start(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Exports Haystack Documents for content to be shown at the start of the
        page.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image object.
        :type transformed_page_img: TransformedImage
        :param section_heirarchy: Section heirarchy for the page.
        :type section_heirarchy: Optional[tuple[int]]
        :return: List of HaystackDocument objects containing the page content.
        :rtype: List[Document]
        """
        self._validate_element_type(element_info)
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "page_location": "start",
            "section_heirarchy": section_heirarchy,
        }
        if self.page_img_order == "before":
            outputs.extend(
                self._export_page_img_docs(element_info, transformed_page_img, meta)
            )
        if self.page_start_text_formats:
            outputs.extend(
                self._export_page_text_docs(
                    element_info, self.page_start_text_formats, meta
                )
            )
        return outputs

    async def convert_page_end(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[Document]:
        """
        Exports Haystack Documents for content to be shown at the end of the
        page.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image object.
        :type transformed_page_img: TransformedImage
        :param section_heirarchy: Section heirarchy for the page.
        :type section_heirarchy: Optional[tuple[int]]
        :return: List of HaystackDocument objects containing the page content.
        :rtype: List[Document]
        """
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "page_location": "end",
            "section_heirarchy": section_heirarchy,
        }
        if self.page_end_text_formats:
            outputs.extend(
                self._export_page_text_docs(
                    element_info, self.page_end_text_formats, meta
                )
            )
        if self.page_img_order == "after":
            outputs.extend(
                self._export_page_img_docs(element_info, transformed_page_img, meta)
            )
        return outputs

    def _export_page_img_docs(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        """
        Exports the page image as a HaystackDocument.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image object.
        :type transformed_page_img: TransformedImage
        :param meta: Metadata to include in the output documents.
        :type meta: Dict[str, Any]
        :return: List of HaystackDocument objects containing the image content.
        :rtype: List[HaystackDocument]
        """
        # Get transformed image, copying over image transformation metadata
        img_bytestream = HaystackByteStream(
            data=pil_img_to_base64(transformed_page_img.image),
            mime_type="image/jpeg",
        )
        meta["rotation_applied"] = transformed_page_img.rotation_applied
        # Create output docs
        img_outputs = list()
        if self.page_img_text_intro:
            page_intro_content = self.page_img_text_intro.format(
                page_number=element_info.start_page_number
            )
        else:
            page_intro_content = None
        img_outputs.append(
            HaystackDocument(
                id=f"{element_info.element_id}_img",
                content=page_intro_content,
                blob=img_bytestream,
                meta=meta,
            )
        )
        return img_outputs

    def _export_page_text_docs(
        self, element_info: ElementInfo, text_formats: List[str], meta: Dict[str, Any]
    ) -> List[HaystackDocument]:
        """
        Exports text documents for the page.

        :param element_info: Element information for the page.
        :type element_info: ElementInfo
        :param text_formats: List of text formats to use.
        :type text_formats: List[str]
        :param meta: Metadata to include in the output documents.
        :type meta: Dict[str, Any]
        :return: List of HaystackDocument objects containing the text content.
        :rtype: List[HaystackDocument]
        """
        output_strings = list()
        for format in text_formats:
            has_format_placeholders = self._format_str_contains_placeholders(
                format, self.expected_format_placeholders
            )
            formatted_text = format.format(page_number=element_info.start_page_number)
            formatted_if_null = format.format(page_number="")
            if formatted_text != formatted_if_null or not has_format_placeholders:
                output_strings.append(formatted_text)
        if output_strings:
            # Join outputs together with new lines
            return [
                HaystackDocument(
                    id=f"{element_info.element_id}_text",
                    content="\n".join(output_strings),
                    meta=meta,
                )
            ]
        return list()

class PageSpanCalculator:
    """
    Calculator for determining the page number of a span based on its position.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    """

    def __init__(self, analyze_result: AnalyzeResult):
        self.page_span_bounds: Dict[int, SpanBounds] = self._get_page_span_bounds(
            analyze_result
        )
        self._doc_end_span =  self.page_span_bounds[max(self.page_span_bounds)].end

    async def determine_span_start_page(self, span_start_offset: int) -> int:
        """
        Determines the page on which a span starts.

        :param span_start_offset: Span starting offset.
        :type span_position: int
        :raises ValueError: Raised when the span_start_offset is greater than
            the last page's end span.
        :return: The page number on which the span starts.
        :rtype: int
        """
        if span_start_offset > self._doc_end_span:
            raise ValueError(
                f"span_start_offset {span_start_offset} is greater than the last page's end span ({self._doc_end_span})."
            )
        page_numbers = [
            k
            for k, v in self.page_span_bounds.items()
            if v.offset <= span_start_offset and v.end >= span_start_offset
        ]
        return min(page_numbers)

    def _get_page_span_bounds(
        self, analyze_result: AnalyzeResult
    ) -> Dict[int, SpanBounds]:
        """
        Gets the span bounds for each page.

        :param analyze_result: AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :raises ValueError: Raised when a gap exists between the span bounds of
            two pages.
        :return: Dictionary with page number as key and tuple of start and end.
        :rtype: Dict[int, SpanBounds]
        """
        page_span_bounds: Dict[int, SpanBounds] = dict()
        page_start_span = 0  # Set first page start to 0
        for page in analyze_result.pages:
            max_page_bound = get_min_and_max_span_bounds(page.spans).end
            max_word_bound = (
                get_min_and_max_span_bounds([page.words[-1].span]).end
                if page.words
                else -1
            )
            max_bound_across_elements = max(max_page_bound, max_word_bound)
            page_span_bounds[page.page_number] = SpanBounds(
                offset=page_start_span,
                end=max_bound_across_elements,
            )
            page_start_span = max_bound_across_elements + 1
        # Check no spans are missed
        last_end_span = -1
        for page_num, span_bounds in page_span_bounds.items():
            if span_bounds.offset != last_end_span + 1:
                raise ValueError(
                    f"Gap exists between span bounds of pages {page_num-1} and {page_num}"
                )
            last_end_span = span_bounds.end

        return page_span_bounds