from datetime import date
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class UserState(TypedDict):
    name: str
    surname: str
    age: int
    birth_date: date
    message: str


def calculate_age(state: UserState) -> dict[str, int]:
    today = date.today()
    age = today.year - state["birth_date"].year

    if (today.month, today.day) < (state["birth_date"].month, state["birth_date"].day):
        age -= 1

    return {"age": age}


def check_drive(state: UserState) -> str:
    return "можно" if state["age"] >= 18 else "нельзя"


def generate_success_message(state: UserState) -> dict[str, str]:
    return {
        "message": f"Поздравляем, {state['name']} {state['surname']}! "
                   f"Вам уже {state['age']} лет и вы можете водить!"
    }


def generate_failure_message(state: UserState) -> dict[str, str]:
    return {
        "message": f"К сожалению, {state['name']} {state['surname']}, "
                   f"вам ещё только {state['age']} лет и вы не можете водить."
    }


graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)
graph.add_node("generate_success_message", generate_success_message)
graph.add_node("generate_failure_message", generate_failure_message)

graph.add_edge(START, "calculate_age")
graph.add_conditional_edges(
    "calculate_age",
    check_drive,
    {
        "можно": "generate_success_message",
        "нельзя": "generate_failure_message",
    }
)
graph.add_edge("generate_success_message", END)
graph.add_edge("generate_failure_message", END)

app = graph.compile()

result = app.invoke(
    {
        "name": "Валерия",
        "surname": "Качановская",
        "birth_date": date.fromisoformat("2003-12-05"),
    }
)
print(result)

