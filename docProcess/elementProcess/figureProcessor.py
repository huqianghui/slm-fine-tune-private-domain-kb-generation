
import itertools
import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentBarcode,
    DocumentFigure,
    DocumentFormula,
    DocumentPage,
)
from haystack.dataclasses import ByteStream as HaystackByteStream
from haystack.dataclasses import Document as HaystackDocument
from PIL.Image import Image as PILImage

from ..imageTools import (
    TransformedImage,
    crop_img,
    get_flat_poly_lists_convex_hull,
    pil_img_to_base64,
    rotate_polygon,
    scale_flat_poly_list,
)
from .elementInfo import ElementInfo
from .elementProcessor import DocumentElementProcessor
from .elementTools import (
    get_element_number,
    get_min_and_max_span_bounds,
    is_span_in_span,
    replace_content_formulas_and_barcodes,
)
from .selectionMarkFormatter import SelectionMarkFormatter

logger = logging.getLogger(__name__)


class DocumentFigureProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentFigure elements.
    """

    expected_elements = [DocumentFigure]

    @abstractmethod
    async def convert_figure(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting figure elements.
        """

class DefaultDocumentFigureProcessor(DocumentFigureProcessor):
    """
    The default processor for DocumentFigure elements. This class provides
    methods for converting DocumentFigure elements into Haystack Document
    objects for both the text and image content of a figure.

    Text format strings can contain placeholders for the following:
    - {figure_number}: The figure number (ordered from the beginning of the
        document).
    - {caption}: The figure caption.
    - {footnotes}: The figure footnotes.
    - {content}: Text content from within the figure itself.

    This processor outputs multiple Documents for each figure:
    1. Before-image text: A list of text content documents that are output
        prior to the image itself.
    2. Image: The image content of the figure, optionally containing text
        content that can give context to the image itself.
    3. After-image text: A list of text content documents that are output
        following the image itself.

    It is recommended to output most of the figure content (captions, content,
    footnotes etc) prior to the image itself.

    :param before_figure_text_formats: A list of text format strings which define
        the content which should be extracted from the element prior to the
        image itself. If None, no text content will be extracted prior to the
        image.
    :type before_figure_text_formats: List[str], optional
    :param output_figure_img: Whether to output a cropped image as part of the
        outputs, defaults to True
    :type output_figure_img: bool
    :param figure_img_text_format: An optional text format str to be attached to
        the Document containing the image.
    :type figure_img_text_format: str, optional
    :param after_figure_text_formats: : A list of text format strings which define
        the content which should be extracted from the element following the
        image itself. If None, no text content will be extracted after the
        image.
    :type after_figure_text_formats: List[str], optional
    """

    expected_format_placeholders = [
        "{figure_number}",
        "{caption}",
        "{footnotes}",
        "{content}",
    ]

    def __init__(
        self,
        before_figure_text_formats: Optional[List[str]] = [
            "**Figure {figure_number} Info**\n",
            "*Figure Caption:* {caption}",
            "*Figure Footnotes:* {footnotes}",
            "*Figure Content:*\n{content}",
        ],
        output_figure_img: bool = True,
        figure_img_text_format: Optional[str] = "\n*Figure Image:*",
        after_figure_text_formats: Optional[List[str]] = None,
        markdown_img_tag_path_or_url:Optional[str] = None
    ):
        self.before_figure_text_formats = before_figure_text_formats
        self.output_figure_img = output_figure_img
        self.figure_img_text_format = figure_img_text_format
        self.after_figure_text_formats = after_figure_text_formats
        self.markdown_img_tag_path_or_url = markdown_img_tag_path_or_url

    async def convert_figure(
        self,
        element_info: ElementInfo,
        transformed_page_img: TransformedImage,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]]
    ) -> List[HaystackDocument]:
        """
        Converts a DocumentFigure element into a list of HaystackDocument
        objects, containing the text and image content for the figure.

        :param element_info: ElementInfo object for the figure.
        :type element_info: ElementInfo
        :param transformed_page_img: Transformed image of the page containing
            the figure.
        :type transformed_page_img: TransformedImage
        :param analyze_result: The AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :param all_formulas: List of all formulas in the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: List of all barcodes in the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: SelectionMarkFormatter object.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: Section heirarchy of the figure.
        :type section_heirarchy: Optional[tuple[int]]
        :return: List of HaystackDocument objects containing the figure content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        page_numbers = list(
            sorted(
                {region.page_number for region in element_info.element.bounding_regions}
            )
        )
        if len(page_numbers) > 1:
            logger.warning(
                f"Figure spans multiple pages. Only the first page will be used. Page numbers: {page_numbers}"
            )
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "section_heirarchy": section_heirarchy,
        }
        if self.before_figure_text_formats:
            outputs.extend(
                await self._export_figure_text(
                    element_info,
                    analyze_result,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_before_figure_text",
                    self.before_figure_text_formats,
                    meta,
                )
            )
        if self.output_figure_img:
            page_element = analyze_result.pages[
                page_numbers[0] - 1
            ]  # Convert 1-based page number to 0-based index
            figure_img = await self.convert_figure_to_img(
                element_info.element, transformed_page_img, page_element
            )
            figure_img_text_docs = await self._export_figure_text(
                element_info,
                analyze_result,
                all_formulas,
                all_barcodes,
                selection_mark_formatter,
                id="NOT_REQUIRED",
                text_formats=[self.figure_img_text_format],
                meta={},
            )
            figure_img_text = (
                figure_img_text_docs[0].content if figure_img_text_docs else ""
            )
            outputs.append(
                HaystackDocument(
                    id=f"{element_info.element_id}_img",
                    content=figure_img_text,
                    blob=HaystackByteStream(
                        data=pil_img_to_base64(figure_img),
                        mime_type="image/jpeg",
                    ),
                    meta=meta,
                )
            )
        if self.after_figure_text_formats:
            outputs.extend(
                await self._export_figure_text(
                    element_info,
                    analyze_result,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_after_figure_text",
                    self.after_figure_text_formats,
                    meta,
                )
            )

        return outputs

    async def _export_figure_text(
        self,
        element_info: ElementInfo,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        id: str,
        text_formats: List[str],
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        """
        Exports the text content of a figure element.

        :param element_info: ElementInfo object for the figure.
        :type element_info: ElementInfo
        :param analyze_result: The AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :param all_formulas: List of all formulas in the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: List of all barcodes in the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: SelectionMarkFormatter object.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param id: ID of the figure element.
        :type id: str
        :param text_formats: List of text formats to use.
        :type text_formats: List[str]
        :param meta: Metadata for the figure element.
        :type meta: Dict[str, Any]
        :return: List of HaystackDocument objects containing the text content.
        :rtype: List[HaystackDocument]
        """
        figure_number_text = await get_element_number(element_info)
        caption_text = await selection_mark_formatter.format_content(
            await replace_content_formulas_and_barcodes(
                element_info.element.caption.content,
                element_info.element.caption.spans,
                all_formulas,
                all_barcodes,
            )
            if element_info.element.caption
            else ""
        )
        if element_info.element.footnotes:
            footnotes_text = await selection_mark_formatter.format_content(
                "\n".join(
                    [
                        await replace_content_formulas_and_barcodes(
                            footnote.content, footnote.spans, all_formulas, all_barcodes
                        )
                        for footnote in element_info.element.footnotes
                    ]
                )
            )
        else:
            footnotes_text = ""
        # Get text content only if it is necessary (it is a slow operation)
        content_text = await selection_mark_formatter.format_content(
            await self._get_figure_text_content(
                element_info.element,
                analyze_result,
                all_formulas,
                all_barcodes,
                selection_mark_formatter,
            )
            if any("{content}" in format for format in text_formats)
            else ""
        )

        output_strings = list()
        for format in text_formats:
            has_format_placeholders = self._format_str_contains_placeholders(
                format, self.expected_format_placeholders
            )
            formatted_text = format.format(
                figure_number=figure_number_text,
                caption=caption_text,
                footnotes=footnotes_text,
                content=content_text,
            )
            formatted_if_null = format.format(
                figure_number="", caption="", footnotes="", content=""
            )
            if formatted_text != formatted_if_null or not has_format_placeholders:
                output_strings.append(formatted_text)
        if output_strings:
            return [
                HaystackDocument(
                    id=id,
                    content="\n".join(output_strings),
                    meta=meta,
                ),
            ]
        return list()

    async def _get_figure_text_content(
        self,
        figure_element: DocumentFigure,
        analyze_result: AnalyzeResult,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
    ) -> str:
        """
        Gets the text content of a figure element. This method automatically
        excludes the caption and footnotes from the text content.

        :param figure_element: Figure element to extract text content from.
        :type figure_element: DocumentFigure
        :param analyze_result: The AnalyzeResult object.
        :type analyze_result: AnalyzeResult
        :param all_formulas: List of all formulas in the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: List of all barcodes in the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: SelectionMarkFormatter object.
        :type selection_mark_formatter: SelectionMarkFormatter
        :return: The text content of the figure element.
        :rtype: str
        """
        # Identify figure content spans
        caption_spans = figure_element.caption.spans if figure_element.caption else []
        footnote_spans = list(
            itertools.chain.from_iterable(
                [footnote.spans for footnote in figure_element.footnotes or []]
            )
        )
        content_spans = [
            span
            for span in figure_element.spans
            if span not in caption_spans and span not in footnote_spans
        ]
        figure_page_numbers = [
            region.page_number for region in figure_element.bounding_regions
        ]

        content_min_max_span_bounds = get_min_and_max_span_bounds(content_spans)

        output_line_strings = list()
        current_line_strings = list()
        matched_spans = list()
        for content_span in content_spans:
            content_min_max_span_bounds = get_min_and_max_span_bounds(content_spans)
            span_perfect_match = False
            for para in analyze_result.paragraphs or []:
                if para.spans[0].offset > content_min_max_span_bounds.end:
                    break
                span_perfect_match = para.spans[0] == content_span
                span_perfect_match = False
                span_checks = [
                    await is_span_in_span(para_span, content_span)
                    for para_span in para.spans
                ]
                if span_perfect_match or all(span_checks):
                    matched_spans.extend(para.spans)
                    if current_line_strings:
                        output_line_strings.append(" ".join(current_line_strings))
                        current_line_strings = list()
                    output_line_strings.append(
                        await selection_mark_formatter.format_content(
                            await replace_content_formulas_and_barcodes(
                                para.content, para.spans, all_formulas, all_barcodes
                            )
                        )
                    )
            if span_perfect_match:
                continue
            for page_number in figure_page_numbers:
                for line in analyze_result.pages[page_number - 1].lines or []:
                    # If we have already matched the span, we can skip the rest of the lines
                    if line.spans[0].offset > content_min_max_span_bounds.end:
                        break
                    # If line is already part of a higher-priority element, skip it
                    check_list = []
                    for matched_span in matched_spans:
                        for line_span in line.spans:
                            check_list.append(await is_span_in_span(line_span, matched_span))
                    if any(check_list):
                        continue
                    
                    span_perfect_match = line.spans[0] == content_span
                    span_checks = [
                        await is_span_in_span(line_span, content_span)
                        for line_span in line.spans
                    ]
                    if span_perfect_match or all(span_checks):
                        matched_spans.extend(line.spans)
                        if current_line_strings:
                            output_line_strings.append(" ".join(current_line_strings))
                            current_line_strings = list()
                        output_line_strings.append(
                            await selection_mark_formatter.format_content(
                                await replace_content_formulas_and_barcodes(
                                    line.content, line.spans, all_formulas, all_barcodes
                                )
                            )
                        )
                if span_perfect_match:
                    continue
                for word in analyze_result.pages[page_number - 1].words or []:
                    # If we have already matched the span, we can skip the rest of the lines
                    if word.span.offset > content_min_max_span_bounds.end:
                        break
                    # If line is already part of a higher-priority element, skip it
                    span_checks = [
                        await is_span_in_span(word.span, matched_span)
                        for matched_span in matched_spans
                    ]
                    if any(span_checks):
                        continue
                    span_perfect_match = word.span == content_span
                    if span_perfect_match or (await is_span_in_span(word.span, content_span)):
                        current_line_strings.append(
                            await selection_mark_formatter.format_content(
                                await replace_content_formulas_and_barcodes(
                                    word.content,
                                    [word.span],
                                    all_formulas,
                                    all_barcodes,
                                )
                            )
                        )
            if current_line_strings:
                output_line_strings.append(" ".join(current_line_strings))
                current_line_strings = list()
        if output_line_strings:
            return "\n".join(output_line_strings)
        return ""

    async def convert_figure_to_img(
        self,
        figure: DocumentFigure,
        transformed_page_img: TransformedImage,
        page_element: DocumentPage,
    ) -> PILImage:
        """
        Converts a figure element to a cropped image.

        :param figure: DocumentFigure element to be converted.
        :type figure: DocumentFigure
        :param transformed_page_img: TransformedImage object of the page
            containing the figure.
        :type transformed_page_img: TransformedImage
        :param page_element: Page object which contains the figure.
        :type page_element: DocumentPage
        :return: a PIL Image of the cropped figure.
        :rtype: PILImage
        """
        di_page_dimensions = (page_element.width, page_element.height)
        pil_img_dimensions = (
            transformed_page_img.orig_image.width,
            transformed_page_img.orig_image.height,
        )
        # Adjust DI page coordinates from inch-based to pixel-based
        unique_pages = set([br.page_number for br in figure.bounding_regions])
        if len(unique_pages) > 1:
            logger.warning(
                f"Figure contains bounding regions across multiple pages ({unique_pages}). Only image content from the first page will be used."
            )
        figure_polygons = [
            br.polygon
            for br in figure.bounding_regions
            if br.page_number == page_element.page_number
        ]
        # Scale polygon coordinates from DI page to PIL image
        scaled_polygons = [
            scale_flat_poly_list(
                figure_polygon,
                di_page_dimensions,
                pil_img_dimensions,
            )
            for figure_polygon in figure_polygons
        ]
        pixel_polygon = get_flat_poly_lists_convex_hull(scaled_polygons)
        # If the page image has been transformed, adjust the DI page bounding boxes
        if transformed_page_img.rotation_applied:
            pixel_polygon = rotate_polygon(
                pixel_polygon,
                transformed_page_img.rotation_applied,
                transformed_page_img.orig_image.width,
                transformed_page_img.orig_image.height,
            )
        return crop_img(
            transformed_page_img.image,
            pixel_polygon,
        )
