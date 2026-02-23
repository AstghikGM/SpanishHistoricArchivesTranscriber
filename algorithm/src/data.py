from typing import TypedDict

class Result(TypedDict):
    file_id: str
    file_name: str
    text: str

Results = list[Result]