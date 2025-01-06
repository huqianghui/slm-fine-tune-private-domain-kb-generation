from langchain_core.prompts import HumanMessagePromptTemplate

userPromptTemplateImageDescriptor = '''
    Describe the image in the page. current page content is as follows:
    ${page_content}
    The image's caption is {image_caption}.
    Then he image content can be described as follows:
'''

user_prompt_image_descriptor = HumanMessagePromptTemplate.from_template(userPromptTemplateImageDescriptor)