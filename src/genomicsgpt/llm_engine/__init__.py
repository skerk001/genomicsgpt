"""
GenomicsGPT LLM Engine — Clinical variant report generation.

Supports two backends:
  - Ollama (free, local): ollama pull llama3:8b
  - Claude API (paid): requires ANTHROPIC_API_KEY

Usage:
    from genomicsgpt.llm_engine import OllamaReportGenerator, generate_report

    # Free — local Llama 3
    gen = OllamaReportGenerator(model="llama3:8b")
    narrative = gen.generate(variant_report)

    # Or one-liner
    narrative = generate_report("BRCA1 c.5266dupC", backend="ollama")
"""

from genomicsgpt.llm_engine.report_generator import (
    ClinicalNarrative,
    OllamaReportGenerator,
    ReportGenerator,
    assemble_evidence,
    build_prompt,
    generate_report,
)

__all__ = [
    "ClinicalNarrative",
    "OllamaReportGenerator",
    "ReportGenerator",
    "assemble_evidence",
    "build_prompt",
    "generate_report",
]