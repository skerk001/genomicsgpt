"""
LLM Narrative Engine — Generates clinical variant interpretation reports.

Takes a VariantReport (with database annotations + ML predictions) and
produces a plain-English clinical narrative using Claude API.

Pipeline:
    VariantReport → Evidence Assembly → Prompt Construction → Claude API → Structured Report
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Report sections ──────────────────────────────────────────────────────────


@dataclass
class ClinicalNarrative:
    """Structured clinical narrative output from the LLM."""
    variant_summary: str = ""        # 1-2 sentence variant description
    classification: str = ""         # Predicted classification + confidence
    evidence_summary: str = ""       # Key evidence supporting the classification
    molecular_mechanism: str = ""    # How the variant affects the protein/gene
    population_data: str = ""        # Allele frequency interpretation
    clinical_implications: str = ""  # Actionable clinical guidance
    limitations: str = ""            # Caveats and uncertainties
    acmg_criteria: str = ""          # Which ACMG criteria are relevant
    full_report: str = ""            # Complete formatted report
    model_used: str = ""
    tokens_used: int = 0
    generation_time_seconds: float = 0.0


# ── Evidence assembly ────────────────────────────────────────────────────────


def assemble_evidence(report) -> dict:
    """
    Extract and structure all available evidence from a VariantReport
    into a clean dict for prompt construction.
    """
    evidence = {
        "variant": {},
        "clinvar": {},
        "population": {},
        "ml_prediction": {},
        "molecular": {},
        "protein_domains": [],
    }

    # Variant identity
    v = report.variant
    evidence["variant"] = {
        "display_name": v.display_name,
        "raw_input": v.raw_input,
        "gene": v.gene_symbol or "Unknown",
        "rs_id": v.rs_id,
        "variant_type": v.variant_type.value if v.variant_type else "unknown",
    }

    if v.transcript:
        evidence["variant"]["transcript"] = v.transcript.transcript_id
        evidence["variant"]["hgvs_c"] = v.transcript.hgvs_c
        evidence["variant"]["hgvs_p"] = v.transcript.hgvs_p
        evidence["variant"]["consequence"] = v.transcript.consequence.value if v.transcript.consequence else "unknown"

    if v.genomic:
        evidence["variant"]["chromosome"] = v.genomic.chrom_normalized
        evidence["variant"]["position"] = v.genomic.position
        evidence["variant"]["ref"] = v.genomic.ref
        evidence["variant"]["alt"] = v.genomic.alt
        evidence["variant"]["assembly"] = v.genomic.assembly.value

    # ClinVar
    if report.clinvar_records:
        records = []
        for r in report.clinvar_records:
            records.append({
                "significance": r.clinical_significance.value,
                "review_status": r.review_status,
                "stars": r.review_stars,
                "condition": r.condition,
                "submitter": r.submitter,
            })
        evidence["clinvar"] = {
            "num_records": len(records),
            "consensus": report.clinvar_consensus.value if report.clinvar_consensus else "None",
            "records": records[:10],  # Limit to top 10 for prompt length
        }

    # Population frequencies
    if report.population_frequencies:
        freqs = []
        for pf in report.population_frequencies:
            freqs.append({
                "population": pf.population,
                "frequency": pf.frequency,
                "allele_count": pf.allele_count,
                "allele_number": pf.allele_number,
            })
        max_freq = report.max_gnomad_frequency
        evidence["population"] = {
            "frequencies": freqs,
            "max_gnomad_af": max_freq,
            "interpretation": _interpret_frequency(max_freq),
        }

    # ML prediction
    if report.pathogenicity_score is not None:
        evidence["ml_prediction"] = {
            "score": report.pathogenicity_score,
            "label": report.pathogenicity_label,
            "confidence": report.prediction_confidence,
            "top_features": dict(sorted(
                report.feature_importances.items(),
                key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0,
                reverse=True,
            )[:5]) if report.feature_importances else {},
        }

    # AlphaMissense
    if report.alphamissense:
        evidence["molecular"]["alphamissense"] = {
            "score": report.alphamissense.score,
            "classification": report.alphamissense.classification,
        }

    # Protein domains
    if report.protein_domains:
        evidence["protein_domains"] = [
            {"name": d.name, "start": d.start, "end": d.end, "source": d.source}
            for d in report.protein_domains[:5]
        ]

    # ACMG evidence
    if report.acmg_evidence:
        evidence["acmg"] = [
            {"criterion": e.criterion.value, "met": e.met, "reason": e.reason}
            for e in report.acmg_evidence
        ]

    return evidence


def _interpret_frequency(freq: Optional[float]) -> str:
    """Interpret allele frequency in clinical context."""
    if freq is None:
        return "Not observed in population databases"
    if freq == 0:
        return "Absent from gnomAD — consistent with rare pathogenic variant"
    if freq < 0.00001:
        return f"Extremely rare (AF={freq:.6f}) — below BA1 threshold"
    if freq < 0.0001:
        return f"Very rare (AF={freq:.5f}) — does not meet BA1"
    if freq < 0.01:
        return f"Rare (AF={freq:.4f}) — may meet BS1 depending on condition"
    if freq < 0.05:
        return f"Low frequency (AF={freq:.3f}) — likely meets BA1 for most conditions"
    return f"Common variant (AF={freq:.2f}) — meets BA1 stand-alone benign"


# ── Prompt construction ──────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a clinical genomics expert assistant generating variant interpretation reports for genetics professionals. Your reports are:

1. **Evidence-based** — every claim is grounded in the provided data
2. **Structured** — follow ACMG/AMP guidelines framework
3. **Clear** — accessible to clinical geneticists, genetic counselors, and oncologists
4. **Honest about uncertainty** — clearly state when evidence is limited

You will receive structured evidence about a genetic variant. Generate a clinical interpretation report with the following sections:

## Variant Summary
One-paragraph overview: what the variant is, which gene, what molecular consequence.

## Classification
State the predicted classification (Pathogenic/Likely Pathogenic/VUS/Likely Benign/Benign) with confidence level. Reference the ML prediction score and ClinVar consensus.

## Evidence Summary
Summarize the key evidence supporting the classification:
- ClinVar submissions and review status
- Population frequency data
- Molecular consequence and predicted functional impact
- Computational predictions (ML ensemble, AlphaMissense if available)

## Molecular Mechanism
Explain HOW this variant likely affects the protein/gene function. Reference the consequence type, protein domain context, and any known functional impact.

## Population Data
Interpret allele frequencies in clinical context. Reference ACMG BA1/BS1 thresholds where applicable.

## Clinical Implications
Actionable guidance: what this means for the patient, recommended follow-up, cascade testing considerations.

## ACMG Criteria
List which ACMG/AMP criteria are applicable based on available evidence. Use standard codes (PVS1, PM2, PP3, etc.) with brief justification.

## Limitations
Honestly state what data is missing and how it limits interpretation confidence.

Keep the total report under 800 words. Use precise clinical language."""


def build_prompt(evidence: dict) -> str:
    """Build the user prompt with all available evidence."""
    sections = []

    sections.append("# Variant Evidence Package\n")

    # Variant identity
    vi = evidence.get("variant", {})
    sections.append("## Variant Identity")
    sections.append(f"- **Display name:** {vi.get('display_name', 'Unknown')}")
    sections.append(f"- **Gene:** {vi.get('gene', 'Unknown')}")
    if vi.get("rs_id"):
        sections.append(f"- **dbSNP:** {vi['rs_id']}")
    if vi.get("hgvs_c"):
        sections.append(f"- **HGVS (cDNA):** {vi['hgvs_c']}")
    if vi.get("hgvs_p"):
        sections.append(f"- **HGVS (protein):** {vi['hgvs_p']}")
    if vi.get("consequence"):
        sections.append(f"- **Consequence:** {vi['consequence']}")
    if vi.get("chromosome"):
        sections.append(f"- **Position:** chr{vi['chromosome']}:{vi.get('position', '?')}")
        sections.append(f"- **Change:** {vi.get('ref', '?')} > {vi.get('alt', '?')}")
    if vi.get("variant_type"):
        sections.append(f"- **Type:** {vi['variant_type']}")

    # ClinVar
    cv = evidence.get("clinvar", {})
    if cv:
        sections.append("\n## ClinVar Evidence")
        sections.append(f"- **Total submissions:** {cv.get('num_records', 0)}")
        sections.append(f"- **Consensus classification:** {cv.get('consensus', 'None')}")
        for i, rec in enumerate(cv.get("records", [])[:5]):
            sections.append(
                f"  - Submission {i+1}: {rec['significance']} "
                f"({rec['review_status']}, {rec['stars']}★) — {rec['condition']}"
            )

    # Population frequencies
    pop = evidence.get("population", {})
    if pop:
        sections.append("\n## Population Frequencies")
        sections.append(f"- **Max gnomAD AF:** {pop.get('max_gnomad_af', 'N/A')}")
        sections.append(f"- **Interpretation:** {pop.get('interpretation', 'N/A')}")
        for pf in pop.get("frequencies", [])[:5]:
            sections.append(f"  - {pf['population']}: {pf['frequency']:.6f}")

    # ML prediction
    ml = evidence.get("ml_prediction", {})
    if ml:
        sections.append("\n## ML Prediction (GenomicsGPT Ensemble)")
        sections.append(f"- **Pathogenicity score:** {ml.get('score', 'N/A')}")
        sections.append(f"- **Predicted label:** {ml.get('label', 'N/A')}")
        sections.append(f"- **Confidence:** {ml.get('confidence', 'N/A')}")
        if ml.get("top_features"):
            sections.append("- **Top contributing features (SHAP):**")
            for feat, val in ml["top_features"].items():
                if isinstance(val, (int, float)):
                    sections.append(f"  - {feat}: {val:.4f}")

    # Molecular context
    mol = evidence.get("molecular", {})
    if mol.get("alphamissense"):
        am = mol["alphamissense"]
        sections.append(f"\n## AlphaMissense")
        sections.append(f"- **Score:** {am['score']}")
        sections.append(f"- **Classification:** {am['classification']}")

    # Protein domains
    domains = evidence.get("protein_domains", [])
    if domains:
        sections.append("\n## Protein Domain Context")
        for d in domains:
            sections.append(f"- {d['name']} ({d['source']}) — positions {d['start']}-{d['end']}")

    # ACMG
    acmg = evidence.get("acmg", [])
    if acmg:
        sections.append("\n## Pre-computed ACMG Criteria")
        for a in acmg:
            status = "MET" if a["met"] else "NOT MET"
            sections.append(f"- **{a['criterion']}** [{status}]: {a['reason']}")

    sections.append("\n---")
    sections.append("Generate a clinical interpretation report based on the evidence above.")

    return "\n".join(sections)


# ── Claude API client ────────────────────────────────────────────────────────


class _BaseReportGenerator:
    """Base class with shared logic for parsing LLM responses."""

    def _parse_sections(self, report_text: str) -> ClinicalNarrative:
        """Extract structured sections from the LLM's markdown response."""
        narrative = ClinicalNarrative()

        section_map = {
            "variant summary": "variant_summary",
            "classification": "classification",
            "evidence summary": "evidence_summary",
            "molecular mechanism": "molecular_mechanism",
            "population data": "population_data",
            "clinical implications": "clinical_implications",
            "limitations": "limitations",
            "acmg criteria": "acmg_criteria",
        }

        # Split on ## headers
        sections = re.split(r'\n##\s+', report_text)

        for section in sections:
            lines = section.strip().split('\n', 1)
            if len(lines) < 2:
                continue
            header = lines[0].lower().strip().rstrip('#').strip()
            content = lines[1].strip()

            for key, attr in section_map.items():
                if key in header:
                    setattr(narrative, attr, content)
                    break

        return narrative

    def _build_narrative(self, report, full_text: str, model: str,
                         tokens: int, start_time: float) -> ClinicalNarrative:
        """Parse response and fill metadata."""
        narrative = self._parse_sections(full_text)
        narrative.full_report = full_text
        narrative.model_used = model
        narrative.tokens_used = tokens
        narrative.generation_time_seconds = time.time() - start_time
        return narrative

    def generate_batch(self, reports: list) -> list[ClinicalNarrative]:
        """Generate narratives for multiple variants."""
        narratives = []
        for i, report in enumerate(reports):
            logger.info(f"Generating report {i+1}/{len(reports)}...")
            narrative = self.generate(report)
            narratives.append(narrative)
            if i < len(reports) - 1:
                time.sleep(0.5)
        return narratives


class ReportGenerator(_BaseReportGenerator):
    """
    Generates clinical variant reports using Claude API.

    Usage:
        generator = ReportGenerator(api_key="sk-ant-...")
        narrative = generator.generate(variant_report)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self.model = model
        self.max_tokens = max_tokens

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    def generate(self, report) -> ClinicalNarrative:
        start_time = time.time()
        evidence = assemble_evidence(report)
        user_prompt = build_prompt(evidence)

        logger.info(f"Calling Claude ({self.model})...")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            full_report = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return ClinicalNarrative(
                full_report=f"Error generating report: {e}",
                generation_time_seconds=time.time() - start_time,
            )

        return self._build_narrative(report, full_report, self.model, tokens_used, start_time)


class OllamaReportGenerator(_BaseReportGenerator):
    """
    Generates clinical variant reports using Ollama (local LLMs — free).

    Requires Ollama running locally: https://ollama.com
    Install a model: ollama pull llama3:latest

    Usage:
        generator = OllamaReportGenerator(model="llama3:latest")
        narrative = generator.generate(variant_report)
    """

    def __init__(
        self,
        model: str = "llama3:latest",
        base_url: str = "http://localhost:11434",
        max_tokens: int = 2000,
        temperature: float = 0.3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Verify Ollama is running
        try:
            import urllib.request
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                available = [m["name"] for m in data.get("models", [])]
                if not any(model in m for m in available):
                    logger.warning(
                        f"Model '{model}' not found. Available: {available}. "
                        f"Run: ollama pull {model}"
                    )
                else:
                    logger.info(f"Ollama connected — model '{model}' available")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {base_url}. "
                f"Make sure Ollama is running (ollama serve). Error: {e}"
            )

    def generate(self, report) -> ClinicalNarrative:
        start_time = time.time()
        evidence = assemble_evidence(report)
        user_prompt = build_prompt(evidence)

        # Combine system + user prompt for Ollama
        full_prompt = f"{SYSTEM_PROMPT}\n\n---\n\n{user_prompt}"

        logger.info(f"Calling Ollama ({self.model})...")
        try:
            import urllib.request

            payload = json.dumps({
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "num_predict": self.max_tokens,
                    "temperature": self.temperature,
                },
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())

            full_report = data.get("response", "")
            tokens_used = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return ClinicalNarrative(
                full_report=f"Error generating report: {e}",
                generation_time_seconds=time.time() - start_time,
            )

        return self._build_narrative(report, full_report, self.model, tokens_used, start_time)


# ── Convenience function ─────────────────────────────────────────────────────


def generate_report(
    variant_string: str,
    api_key: Optional[str] = None,
    backend: str = "ollama",
    model: Optional[str] = None,
) -> ClinicalNarrative:
    """
    One-liner: parse variant → aggregate data → generate clinical report.

    Args:
        variant_string: Any variant format (e.g., "BRCA1 c.5266dupC", "rs80357906")
        api_key: Anthropic API key (only needed for backend="claude")
        backend: "ollama" (free, local) or "claude" (paid API)
        model: Model name. Defaults to "llama3:latest" for Ollama, "claude-sonnet-4-20250514" for Claude.

    Returns:
        ClinicalNarrative with structured clinical report.

    Usage:
        # Free (requires Ollama running locally)
        report = generate_report("BRCA1 c.5266dupC", backend="ollama")

        # Paid (requires Anthropic API key)
        report = generate_report("BRCA1 c.5266dupC", backend="claude", api_key="sk-ant-...")
    """
    from genomicsgpt.data_aggregator.aggregator import quick_lookup

    # Step 1: Parse + aggregate from databases
    variant_report = quick_lookup(variant_string)

    # Step 2: Generate narrative
    if backend == "claude":
        generator = ReportGenerator(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    else:
        generator = OllamaReportGenerator(model=model or "llama3:latest")

    narrative = generator.generate(variant_report)
    return narrative
