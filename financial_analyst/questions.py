"""Small, explicit question grammar for the first fixture iteration."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class QuestionPlan:
    operation: str
    include_yoy: bool = False


class UnsupportedQuestion(ValueError):
    pass


def plan_question(question: str, company: str, period: str) -> QuestionPlan:
    """Accept complete supported questions; never answer only a recognized subset."""
    text = question.casefold().strip()
    if not text or not company.strip() or not period.strip():
        raise UnsupportedQuestion("Question, company, and period must be explicit.")

    mentioned_periods = re.findall(r"\b[1-4]qfy(?:\d{4}|\d{2})\b", text)
    if any(label != period.casefold() for label in mentioned_periods):
        raise UnsupportedQuestion("The question mentions a different reporting period.")

    company_pattern = r"(?<!\w)" + re.escape(company.casefold()) + r"(?:['’]s)?(?!\w)"
    text = re.sub(company_pattern, "", text)
    text = re.sub(r"(?<!\w)" + re.escape(period.casefold()) + r"(?!\w)", "", text)
    text = re.sub(r"broker['’]s", "broker", text)
    text = re.sub(r"[?,.]", "", text)
    text = " ".join(text.split())

    estimate = r"(?:the )?(?:broker )?estimate"
    comparison = [
        rf"compare (?:the )?revenue actual (?:versus|vs|against) {estimate}",
        rf"did (?:the )?revenue beat {estimate}(?: and by how much)?",
        rf"by how much did (?:the )?revenue (?:exceed|beat|miss) {estimate}",
        rf"what (?:was|is) (?:the )?revenue (?:beat|variance)(?: versus {estimate})?",
    ]
    yoy_suffix = r" and what (?:was|is) (?:the )?yoy growth"
    growth_suffix = r" and explain (?:the |its )?(?:revenue )?growth"
    for pattern in comparison:
        if re.fullmatch(pattern, text):
            return QuestionPlan("compare")
        if re.fullmatch(pattern + yoy_suffix, text):
            return QuestionPlan("compare", include_yoy=True)
        if re.fullmatch(pattern + growth_suffix, text):
            return QuestionPlan("combined")

    if re.fullmatch(r"what (?:was|is) (?:the )?(?:revenue )?yoy growth", text):
        return QuestionPlan("compare", include_yoy=True)
    if re.fullmatch(r"(?:why did (?:the )?revenue grow|what drove (?:the )?revenue growth|explain (?:the )?revenue growth)", text):
        return QuestionPlan("growth")
    if re.fullmatch(rf"why did (?:the )?revenue (?:beat|exceed) {estimate}", text):
        return QuestionPlan("beat_attribution")

    raise UnsupportedQuestion(
        "This fixture iteration cannot answer that complete question. "
        "Use a supported revenue comparison or growth question, or the explicit commands."
    )
