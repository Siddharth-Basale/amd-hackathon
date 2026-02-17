"""
Entity and relation extraction utilities.

This module provides a thin wrapper around an LLM to pull structured
entities and relations out of chunk text.  The design keeps the interface
simple so the extractor can be swapped between Ollama, OpenAI, or any
LangChain-compatible chat model.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.language_models.chat_models import BaseChatModel
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None  # type: ignore
    BaseChatModel = Any  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class ExtractedEntity:
    """Represents a detected entity within a chunk of text."""

    name: str
    type: str
    description: str = ""
    aliases: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)


@dataclass
class ExtractedRelation:
    """Represents a relation/edge between two entities."""

    source: str
    relation: str
    target: str
    evidence: str = ""


@dataclass
class ExtractionResult:
    """Container for entities and relations produced by extraction."""

    entities: List[ExtractedEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    raw_response: Optional[Dict[str, Any]] = None


DEFAULT_PROMPT = """You are an information extraction assistant. Analyse the passage
and return structured JSON describing named entities and relations. Follow the schema:

{{
  "entities": [
     {{
        "name": "...",
        "type": "PERSON | ORGANIZATION | LOCATION | EVENT | OTHER",
        "description": "...",
        "aliases": ["..."],
        "source_ids": ["optional chunk ids if provided"]
     }}
  ],
  "relations": [
     {{
        "source": "entity name",
        "relation": "verb or relation phrase",
        "target": "entity name",
        "evidence": "short quote supporting the relation"
     }}
  ]
}}

Rules:
- If nothing is found, return {{"entities": [], "relations": []}}.
- Use entity names exactly as they appear in the text.
- Prefer concise descriptions taken from the passage.
- Relation evidence should be at most one sentence.

Passage:
\"\"\"{text}\"\"\""""


class EntityExtractor:
    """High-level entity & relation extractor."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        prompt_template: str = DEFAULT_PROMPT,
    ) -> None:
        self.llm = llm or self._build_default_llm(model, temperature)
        self.prompt_template = prompt_template

    def _build_default_llm(
        self, model: str, temperature: float
    ) -> BaseChatModel:
        if ChatOpenAI is None:
            raise RuntimeError(
                "ChatOpenAI (langchain-openai) is required for the default "
                "entity extractor. Install it with `pip install langchain-openai` "
                "or provide a custom LLM instance."
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
        )

    def build_prompt(self, text: str, chunk_id: Optional[str] = None) -> str:
        """Render the extraction prompt."""
        preamble = (
            f"Chunk ID: {chunk_id}\n\n" if chunk_id else ""
        )
        return self.prompt_template.format(text=preamble + text)

    def parse_response(self, response_text: str) -> ExtractionResult:
        """Convert the LLM response into structured entities and relations."""
        def _load_json(text: str) -> Dict[str, Any]:
            return json.loads(text)

        try:
            payload = _load_json(response_text)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse extractor response as JSON: %s", exc)
            candidate = self._extract_json_substring(response_text)
            if candidate:
                try:
                    payload = _load_json(candidate)
                except json.JSONDecodeError as inner_exc:
                    logger.warning("Recovery attempt failed: %s", inner_exc)
                    return ExtractionResult(raw_response={"error": str(inner_exc), "text": response_text})
            else:
                return ExtractionResult(raw_response={"error": str(exc), "text": response_text})

        entities_payload = payload.get("entities", []) or []
        relations_payload = payload.get("relations", []) or []

        entities = [
            ExtractedEntity(
                name=item.get("name", "").strip(),
                type=item.get("type", "").strip() or "OTHER",
                description=item.get("description", "").strip(),
                aliases=[alias.strip() for alias in item.get("aliases", []) if alias],
                source_ids=[src.strip() for src in item.get("source_ids", []) if src],
            )
            for item in entities_payload
            if item.get("name")
        ]

        relations = [
            ExtractedRelation(
                source=item.get("source", "").strip(),
                relation=item.get("relation", "").strip(),
                target=item.get("target", "").strip(),
                evidence=item.get("evidence", "").strip(),
            )
            for item in relations_payload
            if item.get("source") and item.get("target") and item.get("relation")
        ]

        return ExtractionResult(entities=entities, relations=relations, raw_response=payload)

    def extract(
        self,
        text: str,
        *,
        chunk_id: Optional[str] = None,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Run the extraction pipeline."""
        if not text.strip():
            return ExtractionResult()

        prompt = self.build_prompt(text, chunk_id)
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.error("Entity extractor LLM invocation failed: %s", exc)
            return ExtractionResult(raw_response={"error": str(exc)})

        response_text = response.content if hasattr(response, "content") else str(response)
        result = self.parse_response(response_text)

        # Attach chunk id metadata for bookkeeping.
        if chunk_id:
            for entity in result.entities:
                if chunk_id not in entity.source_ids:
                    entity.source_ids.append(chunk_id)

        if extra_context:
            logger.debug("Extractor extra context: %s", extra_context)

        return result

    @staticmethod
    def _extract_json_substring(text: str) -> Optional[str]:
        stack = []
        start = None
        for idx, char in enumerate(text):
            if char == "{":
                if not stack:
                    start = idx
                stack.append(char)
            elif char == "}":
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidate = text[start:idx + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            continue
        return None
