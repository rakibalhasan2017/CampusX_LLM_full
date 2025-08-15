from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate(
    input_variables=["paper", "explanation_type", "length_type"],
    template="""
   please summarize the {paper} for a {explanation_type} audience in a {length_type} manner.
    """
)
