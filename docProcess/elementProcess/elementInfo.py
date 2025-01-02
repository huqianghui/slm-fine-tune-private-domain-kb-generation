
from dataclasses import dataclass
from typing import List, Union

from azure.ai.documentintelligence.models import (
    Document,
    DocumentBarcode,
    DocumentFigure,
    DocumentFormula,
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


@dataclass
class SpanBounds:
    """
    Dataclass representing the outer bounds of a span.
    """

    offset: int
    end: int

    def __hash__(self):
        return hash((self.offset, self.end))



@dataclass
class ElementInfo:
    """
    Dataclass containing information about a document element.
    """

    element_id: str
    element: Union[
        DocumentSection,
        DocumentPage,
        DocumentParagraph,
        DocumentLine,
        DocumentWord,
        DocumentFigure,
        DocumentTable,
        DocumentFormula,
        DocumentBarcode,
        DocumentKeyValuePair,
        Document,
        DocumentSelectionMark
    ]
    full_span_bounds: SpanBounds
    spans: List[DocumentSpan]
    start_page_number: int
    section_heirarchy_incremental_id: tuple[int]