import logging
from typing import TypedDict
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

from langgraph.constants import START, END
from langgraph.graph import StateGraph

from dotenv import load_dotenv
import os


load_dotenv()
logger = logging.getLogger(__name__)


llm = ChatOpenAI(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("API_KEY")
)


class ChatState(TypedDict):
    messages: list[BaseMessage]
    should_continue: bool


class ContinueDecision(TypedDict):
    should_continue: bool


decision_llm = llm.with_structured_output(ContinueDecision, method="function_calling")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    before_sleep=lambda retry_state: print(f"Попытка {retry_state.attempt_number} неудачна, повторяю...")
)
def call_llm(messages: list[BaseMessage]) -> AIMessage:
    return llm.invoke(messages)


def user_input_node(state: ChatState) -> dict:
    user_input = input("Вы: ")
    new_messages = state["messages"] + [HumanMessage(content=user_input)]

    return {"messages": new_messages, "should_continue": True}


def check_conversation_end_node(state: ChatState) -> dict[str, bool]:
    last_user_message = state["messages"][-1]

    decision = decision_llm.invoke([
        SystemMessage(
            content="Определи, хочет ли пользователь завершить разговор на основе его последнего "
            "сообщения. Учитывай прощания, благодарности с намерением уйти, явные просьбы "
            "закончить. Обычные вопросы или продолжение темы — это НЕ завершение."
            ),
        HumanMessage(content=f"Сообщение пользователя: {last_user_message.content}")
    ])

    return {"should_continue": decision["should_continue"]}


def llm_response_node(state: ChatState) -> dict:
    try:
        response = call_llm(state["messages"])
        msg_content = response.content
    except Exception:
        msg_content = "Извините, произошла ошибка. Попробуйте ещё раз."

    print(f"ИИ: {msg_content}")
    return {"messages": state["messages"] + [AIMessage(content=msg_content)]}


def should_continue_node(state: ChatState) -> str:
    return "continue" if state.get("should_continue", True) else "end"


def farewell_node(state: ChatState) -> dict:
    response = llm.invoke(
        state["messages"] + [
            SystemMessage(content="Пользователь завершает разговор. Попрощайся тепло и коротко, в одном предложении.")
        ]
    )
    msg_content = response.content
    print(f"ИИ: {msg_content}")

    new_messages = state["messages"] + [AIMessage(content=msg_content)]
    return {"messages": new_messages}


graph = StateGraph(ChatState)

graph.add_node("user_input_node", user_input_node)
graph.add_node("check_conversation_end_node", check_conversation_end_node)
graph.add_node("llm_response_node", llm_response_node)
graph.add_node("farewell_node", farewell_node)

graph.add_edge(START, "user_input_node")
graph.add_edge("user_input_node", "check_conversation_end_node")
graph.add_conditional_edges(
    "check_conversation_end_node",
    should_continue_node,
    {
        "continue": "llm_response_node",
        "end": "farewell_node"
    }
)
graph.add_edge("llm_response_node", "user_input_node")
graph.add_edge("farewell_node", END)

app = graph.compile()


if __name__ == "__main__":
    print("Добро пожаловать в чат с ИИ!")
    print("-" * 50)

    initial_state = {
        "messages": [
            SystemMessage(
                content="Ты дружелюбный помощник. Отвечай коротко и по делу."
                        "Если пользователь прощается или хочет завершить разговор — тепло попрощайся в поле reply "
                        "и отметь это в should_continue."
            )
        ],
        "should_continue": True
    }

    try:
        final_state = app.invoke(initial_state)

        print("-" * 50)
        print("Чат завершён. До свидания!")
        print(f"Всего сообщений в диалоге: {len(final_state['messages'])}")

    except KeyboardInterrupt:
        print("\n\nЧат прерван пользователем (Ctrl+C)")
    except Exception as e:
        print(f"\nОшибка в работе чата: {e}")