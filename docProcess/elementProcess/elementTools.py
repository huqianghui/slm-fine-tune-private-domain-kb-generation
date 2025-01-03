import re
from typing import List

from azure.ai.documentintelligence.models import (
    DocumentBarcode,
    DocumentBarcodeKind,
    DocumentFormula,
    DocumentSpan,
)
from pylatexenc.latex2text import LatexNodes2Text

from .elementInfo import ElementInfo, SpanBounds

LATEX_NODES_TO_TEXT = LatexNodes2Text()

async def get_element_number(element_info: ElementInfo) -> str:
    return str(int(element_info.element_id.split("/")[-1]) + 1)

async def latex_to_text(latex_str: str) -> str:
    """
    Converts a string containing LaTeX to plain text.

    :param latex_str: The LaTeX string to convert.
    :type latex_str: str
    :return: The plain text representation of the LaTeX string.
    :rtype: str
    """
    return LATEX_NODES_TO_TEXT.latex_to_text(latex_str).strip()

async def is_span_in_span(
    span: DocumentSpan,
    parent_span: DocumentSpan,
) -> bool:
    """
    Checks if a given span is contained by a parent span.

    :param span: Span to check.
    :type span: DocumentSpan
    :param parent_span: Parent span to check against.
    :type parent_span: DocumentSpan
    :return: Whether the span is contained by the parent span.
    :rtype: bool
    """
    return span.offset >= parent_span.offset and (span.offset + span.length) <= (
        parent_span.offset + parent_span.length
    )

async def get_barcodes_in_spans(
    all_barcodes: List[DocumentBarcode],
    spans: List[DocumentSpan],
) -> List[DocumentBarcode]:
    """
    Get all barcodes contained within a list of given spans.

    :param all_barcodes: A list of all barcodes in the document.
    :type all_barcodes: List[DocumentBarcode]
    :param span: The span to match.
    :type span: DocumentSpan
    :return: The barcodes contained within the given span.
    :rtype: List[DocumentBarcode]
    """
    matching_barcodes = list()
    for span in spans:
        matching_barcodes.extend(
            [barcode for barcode in all_barcodes if is_span_in_span(barcode.span, span)]
        )
    return matching_barcodes

async def substitute_content_formulas(
    content: str, matching_formulas: list[DocumentFormula]
) -> str:
    """
    Substitute formulas in the content with their actual values.

    :param content: The content to substitute formulas in.
    :type content: str
    :param matching_formulas: The formulas to substitute.
    :type matching_formulas: list[DocumentFormula]
    :returns: The content with the formulas substituted.
    :rtype: str
    """
    # Get all string indices of :formula: in the content
    match_bounds = [
        (match.start(), match.end()) for match in re.finditer(r":formula:", content)
    ]
    if not len(match_bounds) == len(matching_formulas):
        raise ValueError(
            "The number of formulas to substitute does not match the number of :formula: placeholders in the content."
        )
    # Regex may throw issues with certain characters (e.g. double backslashes), so do the replacement manually
    new_content = ""
    last_idx = 0
    for match_bounds, matching_formula in zip(match_bounds, matching_formulas):
        if match_bounds[0] == last_idx:
            new_content += latex_to_text(matching_formula.value)
            last_idx = match_bounds[1]
        else:
            new_content += content[last_idx : match_bounds[0]]
            new_content += latex_to_text(matching_formula.value)
            last_idx = match_bounds[1]
    new_content += content[last_idx:]
    return new_content

async def substitute_content_barcodes(
    content: str, matching_barcodes: list[DocumentBarcode]
) -> str:
    """
    Substitute barcodes in the content with their actual values.

    :param content: The content to substitute barcodes in.
    :type content: str
    :param matching_barcodes: The barcodes to substitute.
    :type matching_barcodes: list[DocumentBarcode]
    :returns: The content with the barcodes substituted.
    :rtype: str
    """
    # Get all string indices of :barcode: in the content
    match_bounds = [
        (match.start(), match.end()) for match in re.finditer(r":barcode:", content)
    ]
    if not len(match_bounds) == len(matching_barcodes):
        raise ValueError(
            "The number of barcodes to substitute does not match the number of :barcode: placeholders in the content."
        )
    # Regex may throw issues with certain characters (e.g. double backslashes), so do the replacement manually
    new_content = ""
    last_idx = 0
    for match_bounds, matching_barcode in zip(match_bounds, matching_barcodes):
        kind_str = (
            matching_barcode.kind.value
            if isinstance(matching_barcode.kind, DocumentBarcodeKind)
            else matching_barcode.kind
        )
        if match_bounds[0] == last_idx:
            new_content += f"*Barcode value:* {matching_barcode.value} (*Barcode kind:* {kind_str})"
            last_idx = match_bounds[1]
        else:
            new_content += content[last_idx : match_bounds[0]]
            new_content += f"*Barcode value:* {matching_barcode.value} (*Barcode kind:* {kind_str})"
            last_idx = match_bounds[1]
    new_content += content[last_idx:]
    return new_content

async def get_formulas_in_spans(
    all_formulas: List[DocumentFormula],
    spans: List[DocumentSpan],
) -> List[DocumentFormula]:
    """
    Get all formulas contained within a list of given spans.

    :param all_formulas: A list of all formulas in the document.
    :type all_formulas: List[DocumentFormula]
    :param span: The span to match.
    :type span: DocumentSpan
    :return: The formulas contained within the given span.
    :rtype: List[DocumentFormula]
    """
    matching_formulas = list()
    for span in spans:
        matching_formulas.extend(
            [formula for formula in all_formulas if is_span_in_span(formula.span, span)]
        )
    return matching_formulas

async def replace_content_formulas_and_barcodes(
    content: str,
    content_spans: List[DocumentSpan],
    all_formulas: List[DocumentFormula],
    all_barcodes: List[DocumentBarcode],
) -> str:
    """
    Replace formulas in the content with their actual values.

    :param content: The content to replace formulas in.
    :type content: str
    :param content_spans: The spans of the content.
    :type content_spans: List[DocumentSpan]
    :param all_formulas: A list of all formulas in the document.
    :type all_formulas: List[DocumentFormula]
    :param all_barcodes: A list of all barcodes in the document.
    :type all_barcodes: List[DocumentBarcode]
    :returns: The content with the formulas and barcdoes replaced.
    :rtype: str
    """
    if ":formula:" in content:
        matching_formulas = await get_formulas_in_spans(all_formulas, content_spans)
        content = await substitute_content_formulas(content, matching_formulas)
    if ":barcode:" in content:
        matching_barcodes = await get_barcodes_in_spans(all_barcodes, content_spans)
        content = await substitute_content_barcodes(content, matching_barcodes)
    return content

def get_min_and_max_span_bounds(
    spans: List[DocumentSpan]
) -> SpanBounds:
    """
    Get the minimum and maximum offsets of a list of spans.

    :param spans: List of spans to get the offsets for.
    :type spans: List[DocumentSpan]
    :return: Tuple containing the minimum and maximum offsets.
    :rtype: SpanBounds
    """
    min_offset = min([span.offset for span in spans])
    max_offset = max([span.offset + span.length for span in spans])
    return SpanBounds(offset=min_offset, end=max_offset)