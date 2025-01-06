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
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from PIL.Image import Image as PILImage

from prompt.systemPromptImageDescription import system_prompt_image_descripter
from prompt.userPromptImageDescription import user_prompt_image_descriptor
from roundRobin.azureOpenAIClientRoundRobin import client_manager

from ..elementProcess.elementInfo import ElementInfo
from ..elementProcess.figureProcessor import DefaultDocumentFigureProcessor
from ..elementProcess.selectionMarkFormatter import SelectionMarkFormatter
from ..imageTools import TransformedImage, pil_img_str_to_png_url

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

            cationText = await self.get_image_caption_text(element_info,selection_mark_formatter,all_formulas,all_barcodes)

            current_workspace_path = os.getenv("CURRENT_WORKSPACE_PATH")
            hash_object = hashlib.md5(element_info.element_id.encode("utf-8")) 
            hash_code = hash_object.hexdigest() 
            file_path = os.path.join(current_workspace_path,'processed_documents','img',f"{hash_code}_img.png")
            figure_img.save(file_path)

            # add the image to the markdown with content as alt text
            figure_img_text_llm = await self._get_image_description_by_LLM(figure_img,page_element,cationText)
            
            if self.markdown_img_tag_path_or_url:
                figure_img_text = f"![]({self.markdown_img_tag_path_or_url}{hash_code}_img.png) \n  the image's description by larage language mode: {figure_img_text_llm} \n"
            else:
                figure_img_text = f"![]({hash_code}_img.png) \n  the image's description by larage language mode: {figure_img_text_llm} \n"

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
    
    async def _get_image_description_by_LLM(self,image:PILImage,page:DocumentPage,cationText:str)->str:
        domain = os.getenv("CURRENT_DOCUMENT_DOMAIN","")

        markdown_lines = []
        for line in page.lines:
            markdown_lines.append(line.content)
        
        all_markdown_content = "\n".join(markdown_lines)

        asyncAzureOpenclient = await client_manager.get_next_client()
        msg_content_list = list()
        msg_content_list.append(ChatCompletionContentPartTextParam(type="text", text=user_prompt_image_descriptor.format(page_content=all_markdown_content,image_caption=cationText).content))
        msg_content_list.append(ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(url=pil_img_str_to_png_url(image),detail="auto")))
        
        message_dict = {"role": "user", "content": msg_content_list}
        userMessageList = ChatCompletionUserMessageParam(message_dict)
        response = await asyncAzureOpenclient.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_ROUND_ROBIN_DEPLOYMENT_NAME","gpt-4o"),
            messages=[
            {
                "role": "system",

                "content": system_prompt_image_descripter.format(domain=domain).content
            },
            userMessageList
            ]
        )
        return response.choices[0].message.content