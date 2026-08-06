import json
import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from dotenv import load_dotenv
import os

from parsers.analysis_parser import review_analysis_prompt, review_parser
from parsers.classification_parser import classification_prompt, classification_parser

load_dotenv()
logger = logging.getLogger(__name__)


llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("API_KEY")
)

classification_chain = classification_prompt | llm | classification_parser
review_analysis_chain = review_analysis_prompt | llm | review_parser


class SystemState(TypedDict):
    messages: list[BaseMessage]
    current_user_input: str
    message_type: str
    should_continue: bool
    analysis_results: list[dict]


def user_input_node(state: SystemState) -> dict:
    user_input = input("Вы: ")

    if user_input.lower() in ["стат", "статистика", "results"]:
        analysis_results = state.get("analysis_results", [])
        if analysis_results:
            print(f"\n📊 Проанализировано отзывов: {len(analysis_results)}")
            sentiments = [r["analysis"]["sentiment"] for r in analysis_results]
            print(
                f"Положительные: {sentiments.count('pos')}, "
                f"Отрицательные: {sentiments.count('neg')}, "
                f"Нейтральные: {sentiments.count('neu')}"
            )
        else:
            print("📊 Пока нет проанализированных отзывов")
        return {"should_continue": True}

    return {
        "current_user_input": user_input,
        "should_continue": True
    }


def classify_message_node(state: SystemState) -> dict:
    user_input = state["current_user_input"]
    try:
        print("🤔 Определяю тип сообщения...")

        result = classification_chain.invoke({"user_input": user_input})

        message_type = result["message_type"]
        confidence = result["confidence"]

        print(f"📝 Тип: {message_type} (уверенность: {confidence:.2f})")

        return {"message_type": message_type}
    except Exception as e:
        print(f"❌ Ошибка классификации: {e}")
        return {"message_type": "question"}


def analyze_review_node(state: SystemState) -> dict:
    user_input = state["current_user_input"]

    try:
        print("🔍 Анализирую отзыв...")

        analysis_result = review_analysis_chain.invoke({"review": user_input})

        full_result = {
            "original_review": user_input,
            "analysis": analysis_result
        }

        analysis_results = state.get("analysis_results", [])
        new_analysis_results = analysis_results + [full_result]

        print("\n" + "=" * 60)
        print("📊 АНАЛИЗ ОТЗЫВА (JSON):")
        print("=" * 60)
        print(json.dumps(full_result, ensure_ascii=False, indent=2))
        print("=" * 60)

        messages = state["messages"]
        new_messages = messages + [
            HumanMessage(content=user_input),
            AIMessage(
                content=f"Отзыв проанализирован: {analysis_result['sentiment']} "
                        f"тональность с уверенностью {analysis_result['confidence']:.2f}"
            )
        ]

        return {
            "messages": new_messages,
            "analysis_results": new_analysis_results
        }

    except Exception as e:
        print(f"❌ Ошибка анализа отзыва: {e}")

        messages = state["messages"]
        new_messages = messages + [
            HumanMessage(content=user_input),
            AIMessage(content="Извините, произошла ошибка при анализе отзыва.")
        ]

        return {"messages": new_messages}


def answer_question_node(state: SystemState) -> dict:
    user_input = state["current_user_input"]

    try:
        print("💬 Отвечаю на вопрос...")

        messages = state["messages"] + [HumanMessage(content=user_input)]
        response = llm.invoke(messages)
        ai_response = response.content

        print(f"🤖 ИИ: {ai_response}")

        new_messages = messages + [AIMessage(content=ai_response)]
        return {"messages": new_messages}

    except Exception as e:
        print(f"❌ Ошибка при ответе: {e}")

        messages = state["messages"] + [
            HumanMessage(content=user_input),
            AIMessage(content="Извините, произошла ошибка при обработке вашего вопроса.")
        ]
        return {"messages": messages}


def route_after_input(state: SystemState) -> str:
    if not state.get("should_continue"):
        return "end"
    if state.get("current_user_input"):
        return "classify"
    return "get_input"


def route_after_classification(state: SystemState) -> str:
    message_type = state.get("message_type", "question")
    return "analyze_review" if message_type == "review" else "answer_question"


def route_continue(state: SystemState) -> str:
    return "get_input" if state.get("should_continue", True) else "end"


graph = StateGraph(SystemState)

graph.add_node("get_input", user_input_node)
graph.add_node("classify", classify_message_node)
graph.add_node("analyze_review", analyze_review_node)
graph.add_node("answer_question", answer_question_node)

graph.add_edge(START, "get_input")
graph.add_conditional_edges(
    "get_input",
    route_after_input,
    {"classify": "classify", "get_input": "get_input", "end": END}
)
graph.add_conditional_edges(
    "classify",
    route_after_classification,
    {"analyze_review": "analyze_review", "answer_question": "answer_question"}
)
graph.add_conditional_edges(
    "analyze_review",
    route_continue,
    {"get_input": "get_input", "end": END}
)
graph.add_conditional_edges(
    "answer_question",
    route_continue,
    {"get_input": "get_input", "end": END}
)

app = graph.compile()

if __name__ == "__main__":
    print("🤖 Умная система: Анализ отзывов + Чат-бот")
    print("Введите отзыв - получите JSON анализ")
    print("Задайте вопрос - получите ответ")
    print("Команды: 'стат' - статистика, 'выход' - завершить")
    print("-" * 60)

    initial_state = {
        "messages": [],
        "current_user_input": "",
        "message_type": "",
        "should_continue": True,
        "analysis_results": []
    }

    try:
        final_state = app.invoke(initial_state)
        print("\n✅ Работа завершена!")
        print(f"📝 Всего сообщений: {len(final_state.get('messages', []))}")
        print(f"📊 Проанализировано отзывов: {len(final_state.get('analysis_results', []))}")

    except KeyboardInterrupt:
        print("\n\n⚠️ Работа прервана (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Ошибка системы: {e}")