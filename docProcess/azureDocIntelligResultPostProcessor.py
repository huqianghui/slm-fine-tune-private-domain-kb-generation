import logging
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Union

from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentFigure,
    DocumentKeyValuePair,
    DocumentLine,
    DocumentPage,
    DocumentParagraph,
    DocumentSection,
    DocumentSelectionMark,
    DocumentSpan,
    DocumentTable,
    DocumentWord,
)
from haystack.dataclasses import Document as HaystackDocument
from PIL.Image import Image as PILImage

from .docIntelligElementTools import (
    ProcessedDocIntelElementDocumentType,
    convert_element_heirarchy_to_incremental_numbering,
    document_span_to_span_bounds,
    get_all_barcodes,
    get_all_formulas,
    get_element_heirarchy_mapper,
    get_element_span_info_list,
    get_processed_di_doc_type,
    order_element_info_list,
)
from .elementProcess.elementTools import is_span_in_span
from .elementProcess.figureProcessor import (
    DefaultDocumentFigureProcessor,
    DocumentFigureProcessor,
)
from .elementProcess.keyValuePairProcessor import (
    DefaultDocumentKeyValuePairProcessor,
    DocumentKeyValuePairProcessor,
)
from .elementProcess.lineProcessor import (
    DefaultDocumentLineProcessor,
    DocumentLineProcessor,
)
from .elementProcess.pageProcessor import (
    DefaultDocumentPageProcessor,
    DocumentPageProcessor,
    PageSpanCalculator,
)
from .elementProcess.paragraphProcessor import (
    DefaultDocumentParagraphProcessor,
    DocumentParagraphProcessor,
)
from .elementProcess.sectionProcessor import (
    DefaultDocumentSectionProcessor,
    DocumentSectionProcessor,
)
from .elementProcess.selectionMarkFormatter import (
    DefaultSelectionMarkFormatter,
    SelectionMarkFormatter,
)
from .elementProcess.tableProcessor import (
    DefaultDocumentTableProcessor,
    DocumentTableProcessor,
)
from .elementProcess.wordProcessor import (
    DefaultDocumentWordProcessor,
    DocumentWordProcessor,
)
from .imageTools import TransformedImage


class DocumentIntelligenceProcessor:
     
    def __init__(self,
        page_processor: DocumentPageProcessor = DefaultDocumentPageProcessor(),
        section_processor: DocumentSectionProcessor = DefaultDocumentSectionProcessor(),
        table_processor: DocumentTableProcessor = DefaultDocumentTableProcessor(),
        figure_processor: DocumentFigureProcessor = DefaultDocumentFigureProcessor(),
        key_value_pair_processor: DocumentKeyValuePairProcessor = DefaultDocumentKeyValuePairProcessor(),
        paragraph_processor: DocumentParagraphProcessor = DefaultDocumentParagraphProcessor(),
        line_processor: DocumentLineProcessor = DefaultDocumentLineProcessor(),
        word_processor: DocumentWordProcessor = DefaultDocumentWordProcessor(),
        selection_mark_formatter: SelectionMarkFormatter = DefaultSelectionMarkFormatter()):

        self._page_processor = page_processor
        self._section_processor = section_processor
        self._table_processor = table_processor
        self._figure_processor = figure_processor
        self._key_value_pair_processor = key_value_pair_processor
        self._paragraph_processor = paragraph_processor
        self._line_processor = line_processor
        self._word_processor = word_processor
        self._selection_mark_formatter = selection_mark_formatter

    def process_analyze_result(
        self,
        analyze_result: AnalyzeResult,
        doc_page_imgs: Optional[Dict[int, PILImage]] = None,
        on_error: Literal["ignore", "raise"] = "ignore",
        break_after_element_idx: Optional[int] = None,
    ) -> List[HaystackDocument]:
        """
        Processes the result of a Document Intelligence analyze operation and
        returns the extracted content as a list of Haystack Documents. This
        output is ready for splitting into separate chunks and/or use with LLMs.

        If image outputs are configured for the page or figure processors, the
        source PDF must be provided using either the `pdf_path` or `pdf_url`
        parameters.

        This process maintains the order of the content as it appears in the
        source document, with the configuration of each separate element
        processor responsible for how the content is processed.

        :param analyze_result: The result of an analyze operation.
        :type analyze_result: Union[AnalyzeResult, FormRecognizerAnalyzeResult]
        :param pdf_path: Local path of the PDF, defaults to None
        :type pdf_path: Union[str, os.PathLike], optional
        :param pdf_url: URL path to PDF, defaults to None
        :type pdf_url: str, optional
        :param on_error: How to handle errors, defaults to "ignore"
        :type on_error: Literal["ignore", "raise"], optional
        :param break_after_element_idx: If provided, this will break the
            processing loop after this many items. defaults to None
        :type break_after_element_idx: int, optional
        :returns: A list of Haystack Documents containing the processed content.
        :rtype: List[HaystackDocument]
        """
        if doc_page_imgs is not None:
            # Check page image IDs match the DI page IDs
            di_page_ids = {page.page_number for page in analyze_result.pages}
            if not doc_page_imgs.keys() == di_page_ids:

                raise ValueError(
                    "doc_page_imgs keys do not match DI page numbers. doc_page_imgs Keys: {}, DI Page Numbers: {}".format(
                        f"({min(doc_page_imgs.keys())} -> {max(doc_page_imgs.keys())})",
                        f"{min(di_page_ids)} -> {max(di_page_ids)}",
                    )
                )
        # Get mapper of element to section heirarchy
        elem_heirarchy_mapper = get_element_heirarchy_mapper(analyze_result)
        section_to_incremental_id_mapper = (
            convert_element_heirarchy_to_incremental_numbering(elem_heirarchy_mapper)
        )

        page_span_calculator = PageSpanCalculator(analyze_result)

        # # Get list of all spans in order
        element_span_info_list = get_element_span_info_list(
            analyze_result, page_span_calculator, section_to_incremental_id_mapper
        )
        ordered_element_span_info_list = order_element_info_list(element_span_info_list)

        all_formulas = get_all_formulas(analyze_result)
        all_barcodes = get_all_barcodes(analyze_result)

        # Create outputs
        full_output_list: List[HaystackDocument] = list()

        transformed_page_imgs: dict[TransformedImage] = dict()  # 1-based page numbers
        current_page_info = None
        current_section_heirarchy_incremental_id = None
        # Keep track of all spans already processed on the page. We will use this to
        # skip low-priority elements that may already be contained in higher-priority
        # elements (e.g. ignoring paragraphs/lines/words that already appear in a table
        # or figure, or lines/words that were already part of a paragraph).
        current_page_priority_spans: List[DocumentSpan] = list()
        unprocessed_element_counter = defaultdict(int)
        # Work through all elements and add to output
        for element_idx, element_info in enumerate(ordered_element_span_info_list):
            try:
                # Skip lower priority elements if their content is already processed as part of a higher-priority element
                if any(
                    isinstance(element_info.element, element_type)
                    for element_type in [
                        DocumentKeyValuePair,
                        DocumentParagraph,
                        DocumentLine,
                        DocumentWord
                    ]
                ):
                    span_already_processed = False
                    for element_span in element_info.spans:
                        if any(
                            is_span_in_span(element_span, processed_span)
                            for processed_span in current_page_priority_spans
                        ):
                            span_already_processed = True
                            break
                    if span_already_processed:
                        continue
                # Output page end outputs if the page has changed
                if (
                    current_page_info is not None
                    and element_info.start_page_number
                    > current_page_info.start_page_number
                ):
                    full_output_list.extend(
                        self._page_processor.convert_page_end(
                            current_page_info,
                            transformed_page_imgs[
                                current_page_info.element.page_number
                            ],
                            current_section_heirarchy_incremental_id,
                        )
                    )
                    # Remove all spans that end before the current page does.
                    current_page_priority_spans = [
                        span
                        for span in current_page_priority_spans
                        if document_span_to_span_bounds(span).offset
                        > current_page_info.full_span_bounds.end
                    ]
                ### Process new elements. The order of element types in this if/else loop matches
                ### the ordering by `ordered_element_span_info_list` and should not be changed.
                if isinstance(element_info.element, DocumentSection):
                    current_section_heirarchy_incremental_id = (
                        element_info.section_heirarchy_incremental_id
                    )
                    full_output_list.extend(
                        self._section_processor.convert_section(
                            element_info, current_section_heirarchy_incremental_id
                        )
                    )
                    # Skip adding section span to priority spans (this would skip all contained content)
                    continue
                elif isinstance(element_info.element, DocumentPage):
                    # Export page image for use by this and other processors (e.g. page and figure processors)
                    transformed_page_imgs[element_info.element.page_number] = (
                        self._page_processor.export_page_img(
                            pdf_page_img=doc_page_imgs[
                                element_info.element.page_number
                            ],
                            di_page=element_info.element,
                        )
                    )
                    current_page_info = element_info
                    full_output_list.extend(
                        self._page_processor.convert_page_start(
                            element_info,
                            transformed_page_imgs[
                                current_page_info.element.page_number
                            ],
                            current_section_heirarchy_incremental_id,
                        )
                    )
                    continue  # Skip adding span to priority spans
                # Process high priority elements with text content
                elif isinstance(element_info.element, DocumentTable):
                    full_output_list.extend(
                        self._table_processor.convert_table(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentFigure):
                    full_output_list.extend(
                        self._figure_processor.convert_figure(
                            element_info,
                            transformed_page_imgs[element_info.start_page_number],
                            analyze_result,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(
                    element_info.element, DocumentSelectionMark):
                    # Skip selection marks as these are processed by each individual processor
                    continue
                elif isinstance(element_info.element, DocumentParagraph):
                    full_output_list.extend(
                        self._paragraph_processor.convert_paragraph(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentLine):
                    full_output_list.extend(
                        self._line_processor.convert_line(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentWord):
                    full_output_list.extend(
                        self._word_processor.convert_word(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                elif isinstance(element_info.element, DocumentKeyValuePair):
                    full_output_list.extend(
                        self._key_value_pair_processor.convert_kv_pair(
                            element_info,
                            all_formulas,
                            all_barcodes,
                            self._selection_mark_formatter,
                            current_section_heirarchy_incremental_id,
                        )
                    )
                # elif isinstance(element.element, Document):
                #     # TODO: Implement processor
                else:
                    unprocessed_element_counter[
                        element_info.element.__class__.__name__
                    ] += 1
                    raise NotImplementedError(
                        f"Processor for {element_info.element.__class__.__name__} is not supported."
                    )
                # Save span start and end for the current element so we can skip lower-priority elements
                current_page_priority_spans.extend(element_info.spans)
                if (
                    break_after_element_idx is not None
                    and element_idx > break_after_element_idx
                ):
                    logging.info(
                        "{} elements processed (break_after_element_idx={}), breaking loop and returning content.".format(
                            element_idx + 1, break_after_element_idx
                        )
                    )
                    break
            except Exception as _e:
                print(
                    f"Error processing element {element_info.element_id} (start_page_number: {element_info.start_page_number}).\nException: {_e}\nElement info: {element_info}"
                )
                if on_error == "raise":
                    raise
        # All content processed, add the final page output and the last chunk
        if current_page_info is not None:
            full_output_list.extend(
                self._page_processor.convert_page_end(
                    current_page_info,
                    transformed_page_imgs[current_page_info.element.page_number],
                    current_section_heirarchy_incremental_id,
                )
            )
        if unprocessed_element_counter:
            print(
                f"Warning: Some elements were not processed due to a lack of support: {dict(unprocessed_element_counter)}"
            )
        return full_output_list

    def merge_adjacent_text_content_docs(
        self,
        chunk_content_list: Union[List[List[HaystackDocument]], List[HaystackDocument]],
        default_text_merge_separator: str = "\n\n",
    ) -> Union[List[HaystackDocument], List[List[HaystackDocument]]]:
        """
        Merges a list of content chunks together so that adjacent text content
        is combined into a single document, with images and other non-text
        content separated into their own documents. This is useful for
        minimizing the number of separate messages that are required when
        sending the content to an LLM.

        Example result:
        Input documents: [text, text, image, text, table, text, image]
        Output documents: [text, image, text, image]

        :param chunk_content_list: A single list of Haystack Documents, or a
            list of lists of Haystack Documents, where
            each sublist contains the content for a single chunk.
        :type chunk_content_list: Union[List[List[HaystackDocument]], List[HaystackDocument]]
        :param default_text_merge_separator: The default separator to use when
            merging text content, defaults to "\n"
        :type default_text_merge_separator: str, optional
        :return: Returns the content in the same format as the input, with
            adjacent text content merged together.
        :rtype: Union[List[HaystackDocument], List[List[HaystackDocument]]]
        """
        is_input_single_list = all(
            isinstance(list_item, HaystackDocument) for list_item in chunk_content_list
        )
        # If a single list is provided, convert it to a list of lists for processing.
        if is_input_single_list:
            temp_chunk_content_list = [chunk_content_list]
        else:
            temp_chunk_content_list = chunk_content_list
        chunk_outputs = list()

        # Join chunks together
        for chunk in temp_chunk_content_list:
            current_text_snippets: List[str] = list()
            current_chunk_content_list = list()
            for content_doc in chunk:
                doc_type = get_processed_di_doc_type(content_doc)
                if doc_type in [
                    ProcessedDocIntelElementDocumentType.TEXT,
                    ProcessedDocIntelElementDocumentType.TABLE,
                ]:
                    # Content is text-only, so add it to the current chunk
                    current_text_snippets.append(content_doc.content)
                    current_text_snippets.append(
                        content_doc.meta.get(
                            "following_separator", default_text_merge_separator
                        )
                    )
                elif doc_type is ProcessedDocIntelElementDocumentType.IMAGE:
                    # We have hit a non-text document.
                    # Join all text in the current chunk into a single str, then add the image bytestream.
                    current_chunk_content_list.append(
                        HaystackDocument(content="".join(current_text_snippets))
                    )
                    current_text_snippets = list()
                    current_chunk_content_list.append(content_doc)

                else:
                    raise ValueError("Unknown processed DI document type.")
            # Add the last chunk
            if current_text_snippets:
                current_chunk_content_list.append(
                    HaystackDocument(content="".join(current_text_snippets))
                )
            chunk_outputs.append(current_chunk_content_list)
        # Return result in same format as input (either list[Document] or list[list[Document]])
        if is_input_single_list:
            return chunk_outputs[0]
        return chunk_outputs

