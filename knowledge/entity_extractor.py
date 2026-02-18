"""
Entity and relation extraction utilities.

This module provides a thin wrapper around an LLM to pull structured
entities and relations out of chunk text. Uses LangChain's with_structured_output
for reliable schema enforcement when supported (e.g. OpenAI).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None  # type: ignore
    Field = None  # type: ignore

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


# Pydantic schema for LangChain with_structured_output
if BaseModel is not None:

    class EntitySchema(BaseModel):
        """Schema for a single extracted entity."""

        name: str = Field(description="Entity name as it appears in the text")
        type: str = Field(
            description="Entity type: a short label (1–2 words) of your choice that best describes the entity's category."
        )
        description: str = Field(default="", description="Concise description from the passage")
        aliases: List[str] = Field(default_factory=list, description="Alternative names")
        source_ids: List[str] = Field(default_factory=list, description="Optional chunk ids")

    class RelationSchema(BaseModel):
        """Schema for a relation between two entities."""

        source: str = Field(description="Source entity name")
        relation: str = Field(description="Verb or relation phrase")
        target: str = Field(description="Target entity name")
        evidence: str = Field(
            default="",
            description=(
                "Exact quote supporting the relation. Use full text—no truncation. "
                "For code snippets with placeholders (e.g. ...), expand to representative concrete values (e.g. chunk_id='doc::chunk::0')."
            ),
        )

    class ExtractionSchema(BaseModel):
        """Schema for the full extraction result."""

        entities: List[EntitySchema] = Field(default_factory=list, description="Named entities")
        relations: List[RelationSchema] = Field(default_factory=list, description="Relations between entities")


EXTRACTION_INTENSITY_INSTRUCTIONS = {
    "minimal": (
        "Extract only the most salient entities and relations. Be conservative—"
        "include only clearly explicit ones. Aim for roughly 1–3 entities and 0–2 relations per passage."
    ),
    "moderate": (
        "Extract a moderate number of entities and relations. Include the most important ones; "
        "infer relations from clear patterns (e.g. lists, tables) but avoid over-extraction. "
        "Aim for roughly 2–5 entities and 1–4 relations per passage."
    ),
}

DEFAULT_PROMPT_TEMPLATE = """You are an information extraction assistant. Analyse the passage
and extract named entities and relations.

Intensity: {intensity_instruction}

Rules:
- Use entity names exactly as they appear in the text.
- Prefer concise descriptions taken from the passage.
- Relation evidence: use the full supporting quote—do not truncate. For code snippets with placeholders like ..., expand them to representative concrete values (e.g. chunk_id='doc::chunk::0') so evidence is informative.
- If nothing is found, return empty entities and relations lists.

Passage:
\"\"\"{text}\"\"\""""


class EntityExtractor:
    """High-level entity & relation extractor using LangChain structured output."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        extraction_intensity: str = "moderate",
        prompt_template: Optional[str] = None,
    ) -> None:
        if BaseModel is None:
            raise RuntimeError(
                "Pydantic is required for structured extraction. "
                "Install with: pip install pydantic"
            )
        intensity = extraction_intensity.lower()
        if intensity not in EXTRACTION_INTENSITY_INSTRUCTIONS:
            raise ValueError(
                f"extraction_intensity must be one of {list(EXTRACTION_INTENSITY_INSTRUCTIONS)}; got {extraction_intensity}"
            )
        self.extraction_intensity = intensity
        self.llm = llm or self._build_default_llm(model, temperature)
        self._structured_llm = self.llm.with_structured_output(
            ExtractionSchema, method="json_schema"
        )
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE

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
        preamble = f"Chunk ID: {chunk_id}\n\n" if chunk_id else ""
        intensity_instruction = EXTRACTION_INTENSITY_INSTRUCTIONS[self.extraction_intensity]
        return self.prompt_template.format(
            intensity_instruction=intensity_instruction,
            text=preamble + text,
        )

    def _pydantic_to_result(self, schema: ExtractionSchema) -> ExtractionResult:
        """Convert Pydantic schema to ExtractionResult."""
        entities = [
            ExtractedEntity(
                name=e.name.strip(),
                type=e.type.strip() or "OTHER",
                description=(e.description or "").strip(),
                aliases=[a.strip() for a in (e.aliases or []) if a],
                source_ids=[s.strip() for s in (e.source_ids or []) if s],
            )
            for e in (schema.entities or [])
            if e.name
        ]
        relations = [
            ExtractedRelation(
                source=r.source.strip(),
                relation=r.relation.strip(),
                target=r.target.strip(),
                evidence=(r.evidence or "").strip(),
            )
            for r in (schema.relations or [])
            if r.source and r.target and r.relation
        ]
        return ExtractionResult(entities=entities, relations=relations)

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
            schema = self._structured_llm.invoke(prompt)
        except Exception as exc:
            logger.error("Entity extractor LLM invocation failed: %s", exc)
            return ExtractionResult(raw_response={"error": str(exc)})

        result = self._pydantic_to_result(schema)

        # Attach chunk id metadata for bookkeeping.
        if chunk_id:
            for entity in result.entities:
                if chunk_id not in entity.source_ids:
                    entity.source_ids.append(chunk_id)

        if extra_context:
            logger.debug("Extractor extra context: %s", extra_context)

        return result
