from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from azure.ai.documentintelligence.models import (
    DocumentBarcode,
    DocumentFormula,
    DocumentTable,
)
from haystack.dataclasses import Document as HaystackDocument

from .elementInfo import ElementInfo
from .elementProcessor import DocumentElementProcessor
from .elementTools import get_element_number, replace_content_formulas_and_barcodes
from .selectionMarkFormatter import SelectionMarkFormatter


class DocumentTableProcessor(DocumentElementProcessor):
    """
    Base processor class for DocumentTable element.
    """

    expected_elements = [DocumentTable]

    @abstractmethod
    async def convert_table(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Method for exporting table elements.
        """


class DefaultDocumentTableProcessor(DocumentTableProcessor):
    """
    The default processor for DocumentTable elements. This class provides
    methods for converting DocumentTable elements into Haystack Document objects
    for both the text and table/dataframe content of a table.

    Text format strings can contain placeholders for the following:
    - {table_number}: The table number (ordered from the beginning of the
        document).
    - {caption}: The table's caption (if available).
    - {footnotes}: The table's footnotes (if available).

    This processor outputs multiple Documents for each figure:
    1. Before-table text: A list of text content documents that are output
        prior to the table itself.
    2. Table: The content of the table as a Haystack Document containing the
        table's content as both a markdown string (content field) and a
        dataframe (dataframe field).
    3. After-table text: A list of text content documents that are output
        following the table itself.

    It is recommended to output most of the table content (captions, content,
    footnotes etc) prior to the table itself.

    :param before_table_text_formats: A list of text format strings which define
        the content which should be extracted from the element prior to the
        table itself. If None, no text content will be extracted prior to the
        image.
    :type before_table_text_formats: List[str], optional
    :param after_table_text_formats: : A list of text format strings which define
        the content which should be extracted from the element following the
        table itself. If None, no text content will be extracted after the
        table.
    :type after_table_text_formats: List[str], optional
    """

    expected_format_placeholders = ["{table_number}", "{caption}", "{footnotes}"]

    def __init__(
        self,
        before_table_text_formats: Optional[List[str]] = [
            "**Table {table_number} Info**\n",
            "*Table Caption:* {caption}",
            "*Table Footnotes:* {footnotes}",
            "*Table Content:*",
        ],
        after_table_text_formats: Optional[List[str]] = None,
    ):
        self.before_table_text_formats = before_table_text_formats
        self.after_table_text_formats = after_table_text_formats

    async def convert_table(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        section_heirarchy: Optional[tuple[int]],
    ) -> List[HaystackDocument]:
        """
        Converts a table element into a list of Haystack Documents containing
        the content of the table.

        :param element_info: Element information for the table.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param section_heirarchy: The section heirarchy of the table.
        :type section_heirarchy: Optional[tuple[int]]
        :return: A list of Haystack Documents containing the table content.
        :rtype: List[HaystackDocument]
        """
        self._validate_element_type(element_info)
        outputs: List[HaystackDocument] = list()
        meta = {
            "element_id": element_info.element_id,
            "element_type": type(element_info.element).__name__,
            "page_number": element_info.start_page_number,
            "section_heirarchy": section_heirarchy,
        }
        if self.before_table_text_formats:
            outputs.extend(
                await self._export_table_text(
                    element_info,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_before_table_text",
                    self.before_table_text_formats,
                    meta,
                )
            )
        # Convert table content
        table_md, table_df = await self._convert_table_content(
            element_info.element, all_formulas, all_barcodes, selection_mark_formatter
        )
        outputs.append(
            HaystackDocument(
                id=f"{element_info.element_id}_table",
                content=table_md,
                dataframe=table_df,
                meta=meta,
            )
        )
        if self.after_table_text_formats:
            outputs.extend(
                await self._export_table_text(
                    element_info,
                    all_formulas,
                    all_barcodes,
                    selection_mark_formatter,
                    f"{element_info.element_id}_after_table_text",
                    self.after_table_text_formats,
                    meta,
                )
            )

        return outputs

    async def _export_table_text(
        self,
        element_info: ElementInfo,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
        id: str,
        text_formats: List[str],
        meta: Dict[str, Any],
    ) -> List[HaystackDocument]:
        """
        Exports the text content for a table element.

        :param element_info: Element information for the table.
        :type element_info: ElementInfo
        :param all_formulas: List of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :param id: ID of the table element.
        :type id: str
        :param text_formats: List of text formats to format with the table's
            content.
        :type text_formats: List[str]
        :param meta: Metadata for the table element.
        :type meta: Dict[str, Any]
        :return: A list of HaystackDocument objects containing the table
            content.
        :rtype: List[HaystackDocument]
        """
        table_number_text = await get_element_number(element_info)
        caption_text = (
            await selection_mark_formatter.format_content(
                await replace_content_formulas_and_barcodes(
                    element_info.element.caption.content,
                    element_info.element.caption.spans,
                    all_formulas,
                    all_barcodes,
                )
            )
            if getattr(element_info.element, "caption", None)
            else ""
        )
        footnotes_text = await selection_mark_formatter.format_content(
            "\n".join(
                [
                    await replace_content_formulas_and_barcodes(
                        footnote.content, footnote.spans, all_formulas, all_barcodes
                    )
                    for footnote in getattr(element_info.element, "footnotes", None)
                    or []
                ]
            )
        )
        output_strings = list()
        for format in text_formats:
            has_format_placeholders = self._format_str_contains_placeholders(
                format, self.expected_format_placeholders
            )
            formatted_text = format.format(
                table_number=table_number_text,
                caption=caption_text,
                footnotes=footnotes_text,
            )
            formatted_if_null = format.format(table_number="", caption="", footnotes="")
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

    async def _convert_table_content(
        self,
        table: DocumentTable,
        all_formulas: List[DocumentFormula],
        all_barcodes: List[DocumentBarcode],
        selection_mark_formatter: SelectionMarkFormatter,
    ) -> tuple[str, pd.DataFrame]:
        """
        Converts table content into both a markdown string and a pandas
        DataFrame.

        :param table: A DocumentTable element as extracted by Azure Document
            Intelligence.
        :type table: DocumentTable
        :param all_formulas: A list of all formulas extracted from the document.
        :type all_formulas: List[DocumentFormula]
        :param all_barcodes: A list of all barcodes extracted from the document.
        :type all_barcodes: List[DocumentBarcode]
        :param selection_mark_formatter: A formatter for selection marks.
        :type selection_mark_formatter: SelectionMarkFormatter
        :return: A tuple of the markdown string and the pandas DataFrame.
        :rtype: tuple[str, pd.DataFrame]
        """
        ### Create pandas DataFrame
        sorted_cells = sorted(
            table.cells, key=lambda cell: (cell.row_index, cell.column_index)
        )
        cell_dict = defaultdict(list)

        num_header_rows = 0
        num_index_cols = 0
        for cell in sorted_cells:
            if cell.kind == "columnHeader":
                num_header_rows = max(num_header_rows, cell.row_index + 1)
            if cell.kind == "rowHeader":
                num_index_cols = max(num_index_cols, cell.column_index + 1)
            # Get text content
            cell_content = await selection_mark_formatter.format_content(
                await replace_content_formulas_and_barcodes(
                    cell.content, cell.spans, all_formulas, all_barcodes
                )
            )
            cell_dict[cell.row_index].append(cell_content)
        table_df = pd.DataFrame.from_dict(cell_dict, orient="index")
        ### Create markdown text version
        table_fmt = "github"
        # Set index
        if num_index_cols > 0:
            table_df.set_index(list(range(0, num_index_cols)), inplace=True)
        index = num_index_cols > 0
        # Set headers
        if num_header_rows > 0:
            table_df = table_df.T
            table_df.set_index(list(range(0, num_header_rows)), inplace=True)
            table_df = table_df.T
            table_md = table_df.to_markdown(index=index, tablefmt=table_fmt)
        else:
            # Table has no header rows. We will create a dummy header row that has empty cells
            table_df_temp = table_df.copy()
            table_df_temp.columns = ["<!-- -->"] * table_df.shape[1]
            table_md = table_df_temp.to_markdown(index=index, tablefmt=table_fmt)
        # Add new line to end of table markdown
        return "\n" + table_md + "\n\n", table_df