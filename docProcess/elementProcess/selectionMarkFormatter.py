from abc import ABC, abstractmethod


class SelectionMarkFormatter(ABC):
    """
    Base formatter class for Selection Mark elements.
    """

    @abstractmethod
    async def format_content(self, content: str) -> str:
        pass

class DefaultSelectionMarkFormatter(SelectionMarkFormatter):
    """
    Default formatter for Selection Mark elements. This class provides a method
    for formatting the content of a Selection Mark element.

    :param selected_replacement: The text used to replace any selected
        text placeholders within the document (anywhere a ':selected:'
        placeholder exists), defaults to "[X]"
    :type selected_replacement: str, optional
    :param unselected_replacement: The text used to replace any unselected
        text placeholders within the document (anywhere an ':unselected:'
        placeholder exists), defaults to "[ ]"
    :type unselected_replacement: str, optional
    """

    def __init__(
        self, selected_replacement: str = "[X]", unselected_replacement: str = "[ ]"
    ):

        self._selected_replacement = selected_replacement
        self._unselected_replacement = unselected_replacement

    async def format_content(self, content: str) -> str:
        """
        Formats text content, replacing any selection mark placeholders with
        the selected or unselected replacement text.

        :param content: Text content to format.
        :type content: str
        :return: Formatted text content.
        :rtype: str
        """
        return content.replace(":selected:", self._selected_replacement).replace(
            ":unselected:", self._unselected_replacement
        )