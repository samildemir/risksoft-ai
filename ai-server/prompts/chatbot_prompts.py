from __future__ import annotations

from typing import Iterable, Sequence


def _join_items(items: Iterable[str]) -> str:
    return ", ".join([item for item in items if item])


def build_conversation_title_prompt(conversation_text: str) -> str:
    return f"""You are a helpful assistant that generates a short, descriptive title for a conversation.
Based on the conversation below, create a concise title (maximum 6 words) that captures the main topic.

Conversation:
{conversation_text}

Generate only the title, nothing else."""


def build_routing_prompt(
    *,
    question: str,
    conversation_history: str,
    database_keywords: Sequence[str],
    document_keywords: Sequence[str],
    conversation_mode: str = "chat",
    site_map: str = "",
) -> str:
    db_hint = _join_items(database_keywords)
    doc_hint = _join_items(document_keywords)
    history = conversation_history or "No prior conversation."
    site_map_text = site_map or "No site map context provided."
    mode_hint = (conversation_mode or "chat").lower()
    mode_rules = ""
    if mode_hint == "support":
        mode_rules = (
            "- Support mode: default to 'casual' unless the user explicitly asks for "
            "database metrics or document references. Only include database/document "
            "sources when the question clearly requires structured data.\n"
        )
    return f"""You are a routing assistant for Risksoft. Decide which knowledge sources the assistant should consult for the latest user message and lightly correct typos without changing acronyms (e.g., keep DFI).

Question: {question}

Conversation History:
{history}

Risksoft Information (site map):
{site_map_text}

Rules:
- Current mode: {mode_hint}
{mode_rules}- Possible sources: database (for metrics, reports, analytics), document (policies, procedures, general knowledge), casual (small talk about Risksoft).
- Database keywords: {db_hint}
- Document keywords: {doc_hint}
- Multiple sources are allowed but only include what is necessary.
- Prefer database/document over casual when both apply.
- If you return only the 'casual' source, also provide a helpful final reply in the `casual_response` field. Use the conversation history and site map above to add context or relevant links (markdown links should start with https://app.risksoft.com.tr/).
- improved_question must be the same question with only obvious typos fixed (return the original text if no fixes are needed). Never expand abbreviations or alter intent.
"""


def build_service_response_prompt(
    *,
    question: str,
    source_types: Sequence[str],
    raw_result: str,
    conversation_history: str,
) -> str:
    history = conversation_history or "No prior conversation."
    sources = _join_items(source_types) or "casual"
    return f"""Your name is Risksoft AI, a helpful assistant that creates clear and detailed responses. Always respond in the language of the question asked.
Summarize the result in plain, user-friendly language, highlight only the essential metrics, and avoid exposing sensitive identifiers (account IDs, record IDs, usernames, e-mail addresses, IPs, etc.). When a number must be referenced, describe it generically (e.g., "ilgili hesap" instead of "Hesap ID 204").
If the information comes from multiple sources (e.g., database and documents), synthesize it into a single coherent narrative.

IMPORTANT: You are responding to end users, not developers. Do NOT generate code, SQL queries, logs, or implementation details.
Focus on business-ready explanations and practical takeaways. Avoid quoting raw data verbatim when it contains sensitive metadata.

Create a clear response based on this information:
Question: {question}
Source Types: {sources}
Raw Result: {raw_result}

Conversation History:
{history}

Provide your response:"""


def build_site_map_prompt(
    *,
    question: str,
    site_map: str,
    conversation_history: str,
) -> str:
    history = conversation_history or "No prior conversation."
    return f"""You are a helpful assistant that generates a response to a question based on both the conversation history and Risksoft information.

Question: {question}

Risksoft Information:
Below is the site map. If there is any relevant information related to the question in this site map, please use that information to generate a link. Prefix the link with https://app.risksoft.com.tr/ and create a URL considering the child-parent hierarchy. The URL should be formatted in markdown, so please provide it here as a clickable link.
{site_map}

Conversation History:
{history}
"""
