import itertools
from collections import defaultdict
from enum import Enum
from typing import Dict, List

from azure.ai.documentintelligence._model_base import Model as DocumentModelBase
from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    Document,
    DocumentBarcode,
    DocumentFigure,
    DocumentFormula,
    DocumentKeyValuePair,
    DocumentLine,
    DocumentList,
    DocumentPage,
    DocumentParagraph,
    DocumentSection,
    DocumentSelectionMark,
    DocumentSpan,
    DocumentTable,
    DocumentWord,
)
from haystack.dataclasses import Document as HaystackDocument

from .elementProcess.elementInfo import ElementInfo, SpanBounds
from .elementProcess.elementTools import get_min_and_max_span_bounds
from .elementProcess.pageProcessor import PageSpanCalculator
from .imageTools import base64_img_str_to_url


class ProcessedDocIntelElementDocumentType(Enum):
    """
    Contains the simplified list of element types that result from processing
    Document Intelligence result.
    """

    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"

async def get_all_formulas(analyze_result: AnalyzeResult) -> List[DocumentFormula]:
    """
    Returns all formulas from the Document Intelligence result.

    :param analyze_result: AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :return: A list of all formulas in the result.
    :rtype: List[DocumentFormula]
    """
    return list(
        itertools.chain.from_iterable(
            page.formulas for page in analyze_result.pages if page.formulas
        )
    )

async def get_all_barcodes(analyze_result: AnalyzeResult) -> List[DocumentBarcode]:
    """
    Returns all barcodes from the Document Intelligence result.

    :param analyze_result: AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :return: A list of all barcodes in the result.
    :rtype: List[DocumentBarcode]
    """
    return list(
        itertools.chain.from_iterable(
            page.barcodes for page in analyze_result.pages if page.barcodes
        )
    )

async def get_processed_di_doc_type(
    processed_di_doc: HaystackDocument,
) -> ProcessedDocIntelElementDocumentType:
    """
    Classifies a processing Document Intelligence document into a simplified
    element type.

    :param processed_di_doc: Document containing processed Document Intelligence
        element content.
    :type processed_di_doc: Document
    :raises ValueError: In case the DI document type is unknown.
    :return: ProcessedDocIntelElementDocumentType Enum.
    :rtype: ProcessedDocIntelElementDocumentType
    """
    if processed_di_doc.dataframe is not None:
        return ProcessedDocIntelElementDocumentType.TABLE
    elif (
        processed_di_doc.blob is not None
        and processed_di_doc.blob.mime_type.startswith("image")
    ):
        return ProcessedDocIntelElementDocumentType.IMAGE
    elif processed_di_doc.content is not None:
        return ProcessedDocIntelElementDocumentType.TEXT
    else:
        raise ValueError(f"Unknown processed DI document type: {processed_di_doc}")

async def document_span_to_span_bounds(
    span: DocumentSpan
) -> SpanBounds:
    """Converts a DocumentSpan object to a SpanBounds object."""
    return SpanBounds(span.offset, span.offset + span.length)

async def order_element_info_list(
    element_span_info_list: List[ElementInfo],
) -> List[ElementInfo]:
    """
    Returns an ordered list of element spans based on the element type, page
    number and the start and end of the span. Ordering is done so that higher
    priority elements have their content processed first, ensuring that any
    lower priority elements are ignored if their content is already contained.

    For example, captions, footnotes and text content of a figure should be
    processed before the raw lines/words (to provide more logical outputs and
    avoid duplication of the content).

    :param element_span_info_list: List of ElementInfo objects to order.
    :type element_span_info_list: List[ElementInfo]
    :return: A reordered list of ElementInfo objects.
    :rtype: List[ElementInfo]
    """
    # Assign a priority to each element type so that higher priority elements are ordered first
    # e.g. pages should appear before any contained content, and paragraphs should always be processed before lines or words
    priority_mapper = {
        DocumentPage: 0,
        DocumentSection: 1,
        DocumentTable: 2,
        DocumentFigure: 2,
        Document: 2,
        DocumentKeyValuePair: 2,
        DocumentParagraph: 3,
        DocumentLine: 4,
        DocumentSelectionMark: 4,
        DocumentWord: 5
    }

    return sorted(
        element_span_info_list,
        key=lambda x: (
            x.start_page_number,
            x.full_span_bounds.offset,
            priority_mapper[type(x.element)],
            0 - x.full_span_bounds.end,
        ),
    )

def get_document_element_spans(
    document_element: DocumentModelBase,
) -> list[DocumentSpan]:
    """
    Get the spans of a document element.

    :param document_element: The document element to get the spans of.
    :type document_element: Model
    :raises NotImplementedError: Raised when the document element does not
        contain a span field.
    :return: The spans of the document element.
    :rtype: list[DocumentSpan]
    """
    if isinstance(document_element, DocumentKeyValuePair):
        return document_element.key.spans + (
            document_element.value.spans if document_element.value else list()
        )
    if hasattr(document_element, "span"):
        return [document_element.span] or list()
    elif hasattr(document_element, "spans"):
        return document_element.spans or list()
    else:
        raise NotImplementedError(
            f"Class {document_element.__class__.__name__} does not contain span field."
        )

async def get_element_span_info_list(
    analyze_result: "AnalyzeResult",
    page_span_calculator: PageSpanCalculator,
    section_to_incremental_id_mapper: Dict[int, tuple[int]],
) -> Dict[DocumentSpan, ElementInfo]:
    """
    Create a mapping of each span start location to the overall position of
    the content. This is used to order the content logically.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :param page_span_calculator: The PageSpanCalculator object to determine the
        page location of a span.
    :type page_span_calculator: PageSpanCalculator
    :param section_to_incremental_id_mapper: A dict containing a mapping of span
        to section heirarchical ID for each object type.
    :type section_to_incremental_id_mapper: Dict[int, tuple[int]]
    :returns: A dictionary with the element ID as key and the order as value.
    """
    # Note: Formula and Barcode elements are not processed by this function as
    # they are processed automatically by the `substitute_content_formulas` and
    # `substitute_content_barcodes` functions.
    # Get info for every unique element
    element_span_info_list = list()
    for attr in [
        "pages",
        "sections",
        "paragraphs",
        "tables",
        "figures",
        "lists",
        "key_value_pairs",
        "documents",
    ]:
        elements: List[
            DocumentPage,
            DocumentSection,
            DocumentParagraph,
            DocumentTable,
            DocumentFigure,
            DocumentList,
            DocumentKeyValuePair,
            Document,
        ] = (
            getattr(analyze_result, attr, list()) or list()
        )
        for elem_idx, element in enumerate(elements):
            spans = get_document_element_spans(element)
            full_span_bounds = get_min_and_max_span_bounds(spans)
            element_span_info_list.append(
                ElementInfo(
                    element_id=f"/{attr}/{elem_idx}",
                    element=element,
                    full_span_bounds=full_span_bounds,
                    spans=spans,
                    start_page_number=await page_span_calculator.determine_span_start_page(
                        full_span_bounds.offset
                    ),
                    section_heirarchy_incremental_id=section_to_incremental_id_mapper.get(
                        attr, {}
                    ).get(
                        elem_idx, None
                    ),
                )
            )
    page_sub_element_counter = defaultdict(int)
    for page_num, page in enumerate(analyze_result.pages):
        for attr in ["lines", "words", "selection_marks"]:
            elements: List[
                DocumentLine,
                DocumentWord,
                DocumentSelectionMark,
            ] = (
                getattr(page, attr) or list()
            )
            for element in elements:
                element_idx = page_sub_element_counter[attr]
                page_sub_element_counter[attr] += 1
                spans = get_document_element_spans(element)
                full_span_bounds = get_min_and_max_span_bounds(spans)
                element_span_info_list.append(
                    ElementInfo(
                        element_id=f"/pages/{page_num}/{attr}/{element_idx}",
                        element=element,
                        full_span_bounds=full_span_bounds,
                        spans=spans,
                        start_page_number=page.page_number,
                        section_heirarchy_incremental_id=None,  # Page elements are not referred to by sections
                    )
                )

    return element_span_info_list

def _convert_section_heirarchy_to_incremental_numbering(
    section_heirarchy: Dict[int, tuple[int]]
) -> Dict[int, tuple[int]]:
    """
    Converts a section heirarchy mapping from a tuple of element IDs to an
    incremental numbering scheme (so that each new section starts from 1).

    :param section_heirarchy: A section heirarchy mapping section ID to a tuple
        of parent section IDs.
    :type section_heirarchy: dict[int, tuple[int]]
    :return: A mapping of section ID to tuple of incremental parent section IDs.
    :rtype: dict[int, tuple[int]]
    """
    # Create a mapping of section level -> tuple of parent section IDs -> dict of section ID: incremental ID
    section_to_section_incremental_id = defaultdict(dict)
    for section_id, parent_section_heirarchy in section_heirarchy.items():
        parent_section_heirarchy = parent_section_heirarchy[:-1]
        parent_level = len(parent_section_heirarchy) - 1
        if parent_level >= 0:
            if (
                parent_section_heirarchy
                not in section_to_section_incremental_id[parent_level]
            ):
                section_to_section_incremental_id[parent_level][
                    parent_section_heirarchy
                ] = {section_id: 1}
            else:
                section_to_section_incremental_id[parent_level][
                    parent_section_heirarchy
                ][section_id] = (
                    len(
                        section_to_section_incremental_id[parent_level][
                            parent_section_heirarchy
                        ]
                    )
                    + 1
                )

    # Now iterate through the mapping to create a final mapping of section ID -> incremental ID
    section_to_incremental_id_mapper = dict()
    for section_id, parent_section_heirarchy in section_heirarchy.items():
        if section_id > 0:
            parent_section_heirarchy = parent_section_heirarchy[:-1]
            incremental_id_tup = tuple()
            # Get iDs for parents
            if len(parent_section_heirarchy) > 1:
                for section_level in range(len(parent_section_heirarchy) - 1):
                    section_incremental_id = section_to_section_incremental_id[
                        section_level
                    ][parent_section_heirarchy[: section_level + 1]][
                        parent_section_heirarchy[section_level + 1]
                    ]
                    incremental_id_tup += (section_incremental_id,)
            # Get current incremental ID
            section_incremental_id = section_to_section_incremental_id[
                len(parent_section_heirarchy) - 1
            ][parent_section_heirarchy][section_id]
            incremental_id_tup += (section_incremental_id,)
            section_to_incremental_id_mapper[section_id] = incremental_id_tup
    return section_to_incremental_id_mapper

async def convert_element_heirarchy_to_incremental_numbering(
    elem_heirarchy_mapper: dict[str, Dict[int, tuple[int]]]
) -> Dict[str, Dict[int, tuple[int]]]:
    """
    Converts a mapping of elements to their parent sections to an incremental
    numbering scheme.

    :param elem_heirarchy_mapper: A mapper containing keys for each element
        type, which inID to a tuple of parent.
    :type elem_heirarchy_mapper: dict[str, dict[int, tuple[int]]]
    :return: A dictionary containing a key for each element type, which each
        contain a mapping of element ID to a tuple of incremental parent section
        IDs.
    :rtype: dict[int, int]
    """
    if "sections" not in elem_heirarchy_mapper:
        # Sections are not present in the heirarchy, meaning no sections were returned from the API. Return an empty dict
        return dict()
    section_to_heirarchy_mapper = elem_heirarchy_mapper["sections"]
    section_to_incremental_id_mapper = (
        _convert_section_heirarchy_to_incremental_numbering(section_to_heirarchy_mapper)
    )

    # Now iterate through all elements and create a mapping of element ID -> incremental heirarchy
    element_to_incremental_id_mapper = defaultdict(dict)
    element_to_incremental_id_mapper["sections"] = section_to_incremental_id_mapper
    for elem_type, elem_heirarchy in elem_heirarchy_mapper.items():
        if elem_type != "sections":
            for elem_id, parent_section_heirarchy in elem_heirarchy.items():
                parent_section = parent_section_heirarchy[-1]
                element_to_incremental_id_mapper[elem_type][elem_id] = (
                    section_to_incremental_id_mapper.get(parent_section, None)
                )

    return dict(element_to_incremental_id_mapper)

async def get_section_heirarchy(
    section_direct_children_mapper: dict,
) -> dict[int, tuple[int]]:
    """
    This function takes a mapper of sections to their direct children and
    returns a full heirarchy of all sections.

    :param section_direct_children_mapper: Mapper of section ID to a list of all
        direct children, including the section itself.
    :type section_direct_children_mapper: dict
    :return: _description_
    :rtype: dict[int, tuple[int]]
    """
    all_section_ids = set(section_direct_children_mapper.keys())
    full_heirarchy = dict()
    missing_ids = set(section_direct_children_mapper.keys())
    while missing_ids:
        next_id = min(missing_ids)
        branch_heirarchy = _get_section_heirarchy(
            section_direct_children_mapper, next_id
        )
        full_heirarchy.update(branch_heirarchy)
        missing_ids = all_section_ids - set(full_heirarchy.keys())
    return dict(sorted(full_heirarchy.items()))

def _get_section_heirarchy(
    section_direct_children_mapper: dict, current_id: int
) -> dict:
    """
    Recursive function to get the heirarchy of a section.

    :param section_direct_children_mapper: Mapper of section ID to a list of all
        direct children, including the section itself.
    :type section_direct_children_mapper: dict
    :param current_id: The ID of the current section to get the heirarchy for.
    :type current_id: int
    :return: A dictionary containing the heirarchy of the current section.
    :rtype: dict
    """
    output = {current_id: (current_id,)}
    for child_id in section_direct_children_mapper[current_id]:
        if current_id != child_id:
            children_heirarchy = _get_section_heirarchy(
                section_direct_children_mapper, child_id
            )
            for parent, children_tup in children_heirarchy.items():
                output[parent] = (current_id,) + children_tup
    return output

async def get_element_heirarchy_mapper(
    analyze_result: "AnalyzeResult",
) -> Dict[str, Dict[int, tuple[int]]]:
    """
    Gets a mapping of each element contained to a heirarchy of its parent
    sections. The result will only contain elements contained by a parent section.

    :param analyze_result: The AnalyzeResult object returned by the
        `begin_analyze_document` method.
    :type analyze_result: AnalyzeResult
    :return: A dictionary containing a key for each element type, which each
        contain a mapping of element ID to a tuple of parent section IDs.
    :rtype: dict[str, dict[int, tuple[int]]]
    """
    if not getattr(analyze_result, "sections", None):
        # Sections do not appear in result (it might be the read model result).
        # Return an empty dict
        return dict()
    # Get section mapper, mapping sections to their direct children
    sections = analyze_result.sections
    section_direct_children_mapper = {
        section_id: [section_id] for section_id in range(len(sections))
    }
    for section_id, section in enumerate(sections):
        for elem in section.elements:
            if "section" in elem:
                child_id = int(elem.split("/")[-1])
                if child_id != section_id:
                    section_direct_children_mapper[section_id].append(child_id)

    # Now recursively work through the mapping to develop a multi-level heirarchy for every section
    section_heirarchy = await get_section_heirarchy(section_direct_children_mapper)

    # Now map each element to the section parents
    elem_to_section_parent_mapper = defaultdict(dict)
    for section_id, section in enumerate(sections):
        for elem in section.elements:
            if "section" not in elem:
                _, elem_type, elem_type_id = elem.split("/")
                elem_to_section_parent_mapper[elem_type][int(elem_type_id)] = (
                    section_heirarchy.get(section_id, ())
                )
    # Return sorted dict
    for key in elem_to_section_parent_mapper:
        elem_to_section_parent_mapper[key] = dict(
            sorted(elem_to_section_parent_mapper[key].items())
        )
    # Now add the section heiarchy
    elem_to_section_parent_mapper["sections"] = section_heirarchy
    return dict(elem_to_section_parent_mapper)

async def convert_processed_di_docs_to_markdown(
    processed_content_docs: List[HaystackDocument],
    default_text_merge_separator: str = "\n",
) -> str:
    """
    Converts a list of processed Document Intelligence documents into a single
    Markdown string. This is useful for rendering the content in a human-readable
    format.

    :param processed_content_docs: List of Documents containing processed
        Document Intelligence element content.
    :type processed_content_docs: List[Document]
    :param default_text_merge_separator: Default text separator to be used when
        joining content, defaults to "\n"
    :type default_text_merge_separator: str
    :raises ValueError: In cases where the processed DI document type is
        unknown.
    :return: A single Markdown string containing the processed content.
    :rtype: str
    """
    output_texts = []
    for content_doc in processed_content_docs:
        # Check if content is italicized or bolded, and add newlines if necessary
        text_content_requires_newlines = content_doc.content is not None and any(
            [
                content_doc.content.startswith("*"),
                content_doc.content.endswith("*"),
                content_doc.content.startswith("_"),
            ]
        )
        doc_type = await get_processed_di_doc_type(content_doc)
        if doc_type in [
            ProcessedDocIntelElementDocumentType.TEXT,
            ProcessedDocIntelElementDocumentType.TABLE,
        ]:
            # Text and table elements can be directly added to the output
            if text_content_requires_newlines:
                output_texts.append("\n\n")
            output_texts.append(content_doc.content)
            if text_content_requires_newlines:
                output_texts.append("\n")
            output_texts.append(
                content_doc.meta.get(
                    "following_separator", default_text_merge_separator
                )
            )
        elif doc_type is ProcessedDocIntelElementDocumentType.IMAGE:
            # We have hit an image element.
            # Hardcode the image into the markdown output so it can be rendered inline.
            if content_doc.content is not None:
                if text_content_requires_newlines:
                    output_texts.append("\n\n")
                output_texts.append(content_doc.content)
                output_texts.append("\n")
            # Pad image with extra newline before and after image
            output_texts.append("\n")
            output_texts.append(
                f"![]({base64_img_str_to_url(content_doc.blob.data.decode(), content_doc.blob.mime_type)})"
            )
            output_texts.append("\n")
            output_texts.append(
                content_doc.meta.get(
                    "following_separator", default_text_merge_separator
                )
            )
        else:
            raise ValueError("Unknown processed DI document type.")
    return "".join(output_texts).strip()

async def convert_processed_di_doc_chunks_to_markdown(
    content_doc_chunks: List[List[HaystackDocument]],
    chunk_prefix: str = "**###### Start of New Chunk ######**",
    default_text_merge_separator: str = "\n",
) -> str:
    """
    Converts a list of lists of processed Document Intelligence documents into
    a single Markdown string, with a prefix shown at the start of each chunk.
    This is useful for rendering the content in a human-readable format.

    :param processed_content_docs: List of lists of Documents containing
        processed  Document Intelligence element content.
    :type processed_content_docs: List[List[Document]]
    :param chunk_prefix: Text prefix to be inserted at the beginning of each
        chunk
    :type chunk_prefix: str
    :param default_text_merge_separator: Default text separator to be used when
        joining content, defaults to "\n"
    :type default_text_merge_separator: str
    :raises ValueError: In cases where the processed DI document type is
        unknown.
    :return: A single Markdown string containing the processed content.
    :rtype: str
    """
    chunked_outputs = list()
    for chunk_docs in content_doc_chunks:
        chunked_outputs.append(chunk_prefix)
        chunked_outputs.append("\n")
        chunked_outputs.append(
            await convert_processed_di_docs_to_markdown(
                chunk_docs, default_text_merge_separator=default_text_merge_separator
            )
        )
    return "\n\n".join(chunked_outputs)