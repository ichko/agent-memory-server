"""Published-protocol LongMemEval judge prompts."""

DEFAULT = (
    "I will give you a question, a correct answer, and a model response. "
    "Answer yes if the response contains the correct answer or is equivalent to it. "
    "Answer no if required information is missing.\n\nQuestion: {question}\n\n"
    "Correct Answer: {answer}\n\nModel Response: {response}\n\n"
    "Is the model response correct? Answer yes or no only."
)
TEMPORAL = DEFAULT.replace(
    "Answer no if required information is missing.",
    "Answer no if required information is missing. Do not penalize an off-by-one "
    "error when the requested answer is a number of days, weeks, or months.",
)
KNOWLEDGE_UPDATE = (
    "Compare the model response with the correct updated answer. A response may "
    "mention older information, but is correct if it clearly gives the required "
    "updated answer."
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\n"
    "Model Response: {response}\n\nAnswer yes or no only."
)
PREFERENCE = (
    "Judge whether the response satisfies the personalized-response rubric. It need "
    "not include every rubric point, but must use the user's personal information "
    "correctly.\n\nQuestion: {question}\n\nRubric: {answer}\n\n"
    "Model Response: {response}\n\nAnswer yes or no only."
)
ABSTENTION = (
    "The question is unanswerable. Judge whether the model correctly identifies that "
    "the requested information is absent or insufficient.\n\nQuestion: {question}\n\n"
    "Explanation: {answer}\n\nModel Response: {response}\n\nAnswer yes or no only."
)

BY_TYPE = {
    "single-session-user": DEFAULT,
    "single-session-assistant": DEFAULT,
    "multi-session": DEFAULT,
    "temporal-reasoning": TEMPORAL,
    "knowledge-update": KNOWLEDGE_UPDATE,
    "single-session-preference": PREFERENCE,
}


def build_judge_prompt(
    *,
    question: str,
    answer: str,
    response: str,
    question_type: str | None,
    abstain: bool,
) -> str:
    template = ABSTENTION if abstain else BY_TYPE.get(question_type or "", DEFAULT)
    return (
        template.replace("{question}", question)
        .replace("{answer}", answer)
        .replace("{response}", response)
    )
