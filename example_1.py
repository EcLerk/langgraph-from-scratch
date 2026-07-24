from datetime import date
from typing import TypedDict

from langgraph.graph import StateGraph, START, END


class UserState(TypedDict):
    name: str
    surname: str
    age: int
    birth_date: date


def calculate_age(state: UserState) -> dict[str, int]:
    today = date.today()
    age = today.year - state["birth_date"].year

    if (today.month, today.day) < (state["birth_date"].month, state["birth_date"].day):
        age -= 1

    return {"age": age}


graph = StateGraph(UserState)

graph.add_node("calculate_age", calculate_age)
graph.add_edge(START, "calculate_age")
graph.add_edge("calculate_age", END)

app = graph.compile()

result = app.invoke(
    {
        "name": "Валерия",
        "surname": "Качановская",
        "birth_date": date.fromisoformat("2003-12-05"),
    }
)
print(result)

