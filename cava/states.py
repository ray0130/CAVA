# Define agent graph
from typing import TypedDict, List

# [TODO]
class VerificationTrace(TypedDict):
    worker_idx: int
    baseline_summary: str
    verification_questions: List[str]
    verification_answers: List[str]
    verified_summary: str


class CoAState(TypedDict):
    query: str
    chunks: List[str]
    i: int
    worker_outputs: List[str]
    verbose: bool
    manager_output: str

    verification_mode: str # "none" | "every" | "every_k"
    verification_k: int
    store_verification_traces: bool
    verification_traces: List[VerificationTrace]
