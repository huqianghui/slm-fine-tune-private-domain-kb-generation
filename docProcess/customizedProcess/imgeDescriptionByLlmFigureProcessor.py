import hashlib
import logging
import os
from typing import List, Optional

from azure.ai.documentintelligence.models import (
    AnalyzeResult,
    DocumentBarcode,
    DocumentFormula,
    DocumentPage,
)
from dotenv import load_dotenv
from haystack.dataclasses import Document as HaystackDocument
from PIL.Image import Image as PILImage

from roundRobin.azureOpenAIClientRoundRobin import client_manager

from ..elementProcess.elementInfo import ElementInfo
from ..elementProcess.figureProcessor import DefaultDocumentFigureProcessor
from ..elementProcess.selectionMarkFormatter import SelectionMarkFormatter
from ..imageTools import TransformedImage

load_dotenv()
logger = logging.getLogger(__name__)

class ImgeDescriptionByLlmFigureProcessor(DefaultDocumentFigureProcessor):
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
            figure_img_context_text = (
                figure_img_text_docs[0].content if figure_img_text_docs else ""
            )

            current_workspace_path = os.getenv("CURRENT_WORKSPACE_PATH")
            hash_object = hashlib.md5(element_info.element_id.encode("utf-8")) 
            hash_code = hash_object.hexdigest() 
            file_path = os.path.join(current_workspace_path,'processed_documents','img',f"{hash_code}_img.png")
            figure_img.save(file_path)

            # add the image to the markdown with content as alt text
            if self.markdown_img_tag_path_or_url:
                figure_img_text = f"![]({self.markdown_img_tag_path_or_url}{hash_code}_img.png)"
            else:
                figure_img_text = f"![]({hash_code}_img.png)"

            outputs.append(
                HaystackDocument(
                    id=f"{element_info.element_id}_img",
                    content=figure_img_text,
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
    
    async def _get_image_description_by_LLM(self,image:PILImage,page:DocumentPage)->str:
        asyncAzureOpenclient = await client_manager.get_next_client()
        pass
