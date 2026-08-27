"""Generic answer-generation instructions for benchmark providers."""

V2_ANSWER_INSTRUCTIONS = (
    "Answer using only the supplied memories. Put the final answer in \\boxed{}. "
    "If the answer cannot be established, output exactly \\boxed{UNKNOWN}. "
    "Do not guess. If the question has a false premise, explain the flaw in \\boxed{}."
)
