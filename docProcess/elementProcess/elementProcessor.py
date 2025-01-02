from abc import ABC
from typing import List

from .elementInfo import ElementInfo


class DocumentElementProcessor(ABC):
    """
    Base processor class for all Document elements extracted by Document
    Intelligence.
    """

    expected_elements = []

    def _validate_element_type(self, element_info: ElementInfo):
        """
        Validates that the Document element contained by the ElementInfo object
        is of the expected type.

        :param element_info: Information about the Document element.
        :type element_info: ElementInfo
        :raises ValueError: If the element of the ElementInfo object is not of
            the expected type.
        """
        if not self.expected_elements:
            raise ValueError("expected_element has not been set for this processor.")

        if not any(
            isinstance(element_info.element, expected_element)
            for expected_element in self.expected_elements
        ):
            raise ValueError(
                f"Element is incorrect type - {type(element_info.element)}. It should be one of `{self.expected_elements}` types."
            )

    def _format_str_contains_placeholders(
        self, format_str: str, expected_placeholders: List[str]
    ) -> bool:
        """
        Returns True if the format string contains any of the expected placeholders.

        :param format_str: Formatting string to check.
        :type format_str: str
        :param expected_placeholders: A list of expected placeholders.
        :type expected_placeholders: List[str]
        :return: Whether the string contains any of the expected placeholders.
        :rtype: bool
        """
        return any([placeholder in format_str for placeholder in expected_placeholders])