from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from models import MessageClassification


classification_parser = JsonOutputParser(pydantic_object=MessageClassification)
classification_prompt = PromptTemplate(
    template="""Определи, является ли это сообщение отзывом о товаре/услуге или обычным вопросом.

ОТЗЫВ - это мнение о товаре, услуге, опыте использования, оценка качества.
ВОПРОС - это запрос информации, общение, просьба о помощи.

Сообщение: {user_input}

{format_instructions}

Верни ТОЛЬКО JSON!""",
    input_variables=["user_input"],
    partial_variables={"format_instructions": classification_parser.get_format_instructions()},
)