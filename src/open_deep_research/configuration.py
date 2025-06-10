import os
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Literal, Optional

from langchain_core.runnables import RunnableConfig


def get_default_feedback_models():
    """Provides the default dictionary for feedback agent models."""
    return {
        "clarity_and_organization": "openai:gpt-4.1-mini",
        "motivation_and_novelty": "openai:gpt-4.1-mini",
        "methodology_and_evidence": "openai:gpt-4.1-mini",
        "technical_and_language_quality": "openai:gpt-4.1-mini",
        "limitations_and_future_work": "openai:gpt-4.1-mini",
    }


@dataclass(kw_only=True)
class Configuration:
    """The configurable fields for the TeachMe Agent."""

    # --- TeachMe Agent Model Configuration ---
    summary_agent_model: str = "openai:gpt-4.1-mini"
    summary_agent_model_kwargs: Optional[Dict[str, Any]] = None

    search_agent_model: str = "openai:gpt-4.1-mini"
    search_agent_model_kwargs: Optional[Dict[str, Any]] = None

    review_agent_model: str = "openai:gpt-4.1-mini"
    review_agent_model_kwargs: Optional[Dict[str, Any]] = None

    # For executive summary and checklist generation
    report_enhancement_agent_model: str = "openai:gpt-4.1-mini"
    report_enhancement_agent_model_kwargs: Optional[Dict[str, Any]] = None

    # --- Feedback Agent Models (Managed by a dictionary) ---
    feedback_agent_models: Dict[str, str] = field(default_factory=get_default_feedback_models)
    feedback_agent_default_model: str = "openai:gpt-4.1-mini"
    feedback_agent_model_kwargs: Optional[Dict[str, Any]] = None

    # --- Search Configuration ---
    max_search_queries_per_topic: int = 5
    max_related_works: int = 5
    max_search_iterations: int = 5
    target_conferences: List[str] = field(default_factory=lambda: ["NeurIPS", "ICLR", "ICML", "CVPR", "ICCV", "ECCV", "ACL", "EMNLP", "NAACL"])

    # --- Review Configuration ---
    review_conference_format: Literal["neurips", "iclr", "icml", "acl"] = "neurips"

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config["configurable"] if config and "configurable" in config else {}
        values: dict[str, Any] = {f.name: os.environ.get(f.name.upper(), configurable.get(f.name)) for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in values.items() if v})
