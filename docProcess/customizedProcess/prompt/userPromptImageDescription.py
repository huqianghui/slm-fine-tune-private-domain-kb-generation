from langchain_core.prompts import HumanMessagePromptTemplate

userPromptTemplateImageDescriptor = '''
# current page content is as follows:
${page_content}

'''

user_prompt_image_descriptor = HumanMessagePromptTemplate.from_template(userPromptTemplateImageDescriptor)