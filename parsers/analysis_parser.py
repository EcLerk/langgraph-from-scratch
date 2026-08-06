from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from models import ReviewAnalysis

review_parser = JsonOutputParser(pydantic_object=ReviewAnalysis)
review_analysis_prompt = PromptTemplate(
    template="""Проанализируй этот отзыв клиента:

    Отзыв: {review}
    
    {format_instructions}
    
    Верни ТОЛЬКО JSON без дополнительных комментариев!
    """,
    input_variables=["review"],
    partial_variables={"format_instructions": review_parser.get_format_instructions()},
)