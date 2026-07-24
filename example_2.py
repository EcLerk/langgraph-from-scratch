from typing import TypedDict

from langgraph.constants import START, END
from langgraph.graph import StateGraph


class UserState(TypedDict):
    age: int
    message: str


def check_age(state: UserState) -> str:
    return "совершеннолетний" if state["age"] >= 18 else "не_совершеннолетний"


def generate_success_message(state: UserState) -> dict:
    """Генерирует сообщение для совершеннолетних"""
    return {"message": f"Вам уже {state['age']} лет и вы можете водить!"}


def generate_failure_message(state: UserState) -> dict:
    """Генерирует сообщение для несовершеннолетних"""
    return {"message": f"Вам ещё только {state['age']} лет и вы не можете водить."}


graph = StateGraph(UserState)

graph.add_node("fake_node", lambda state: state)
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("generate_failure_message", generate_failure_message)

graph.add_edge(START, "fake_node")
graph.add_conditional_edges(
    "fake_node",
    check_age,
    {
        "совершеннолетний": "generate_success_message",
        "не_совершеннолетний": "generate_failure_message",
    }
)
graph.add_edge("generate_success_message", END)
graph.add_edge("generate_failure_message", END)

app = graph.compile()

# Тест для несовершеннолетнего
result_minor = app.invoke({"age": 17})
print("Результат для 17 лет:", result_minor)

# Тест для совершеннолетнего
result_adult = app.invoke({"age": 25})
print("Результат для 25 лет:", result_adult)
