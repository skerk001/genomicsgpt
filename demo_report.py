#!/usr/bin/env python3
"""
GenomicsGPT LLM Demo — Generate a clinical variant report.

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY="sk-ant-..."

    # Generate a report
    python demo_report.py "BRCA1 c.5266dupC"
    python demo_report.py "rs80357906"
    python demo_report.py "BRAF V600E"

    # Or run without args for a default demo
    python demo_report.py
"""

import sys
import os
import time

# Add src to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def demo_with_mock_data(variant_name: str = "BRCA1 c.5266dupC"):
    """
    Demo using mock VariantReport data (no API calls to ClinVar/Ensembl needed).
    Shows the LLM narrative generation in isolation.
    """
    from genomicsgpt.models import (
        ParsedVariant, TranscriptVariant, GenomicPosition,
        ClinVarRecord, PopulationFrequency, VariantReport,
        Assembly, VariantType, MolecularConsequence, ClinicalSignificance,
    )
    from genomicsgpt.llm_engine.report_generator import (
        ReportGenerator, assemble_evidence, build_prompt,
    )

    # ── Build a realistic mock VariantReport ──
    # BRCA1 c.5266dupC (a well-known pathogenic frameshift variant)
    variant = ParsedVariant(
        genomic=GenomicPosition(
            chromosome="17",
            position=43057051,
            ref="C",
            alt="CC",
            assembly=Assembly.GRCH38,
        ),
        transcript=TranscriptVariant(
            transcript_id="NM_007294.4",
            gene_symbol="BRCA1",
            hgvs_c="c.5266dupC",
            hgvs_p="p.Gln1756ProfsTer74",
            exon=20,
            consequence=MolecularConsequence.FRAMESHIFT,
        ),
        rs_id="rs80357906",
        gene_symbol="BRCA1",
        variant_type=VariantType.INSERTION,
        raw_input=variant_name,
    )

    report = VariantReport(
        variant=variant,
        clinvar_records=[
            ClinVarRecord(
                variation_id=17661,
                clinical_significance=ClinicalSignificance.PATHOGENIC,
                review_status="reviewed by expert panel",
                review_stars=3,
                condition="Hereditary breast and ovarian cancer syndrome",
                submitter="ENIGMA consortium",
            ),
            ClinVarRecord(
                variation_id=17661,
                clinical_significance=ClinicalSignificance.PATHOGENIC,
                review_status="criteria provided, multiple submitters, no conflicts",
                review_stars=2,
                condition="Hereditary cancer-predisposing syndrome",
                submitter="GeneDx",
            ),
            ClinVarRecord(
                variation_id=17661,
                clinical_significance=ClinicalSignificance.PATHOGENIC,
                review_status="criteria provided, single submitter",
                review_stars=1,
                condition="Breast-ovarian cancer, familial 1",
                submitter="Invitae",
            ),
        ],
        population_frequencies=[
            PopulationFrequency(population="gnomAD_NFE", frequency=0.0000041, allele_count=5, allele_number=121860),
            PopulationFrequency(population="gnomAD_AFR", frequency=0.0, allele_count=0, allele_number=24042),
            PopulationFrequency(population="gnomAD_EAS", frequency=0.0, allele_count=0, allele_number=19948),
            PopulationFrequency(population="gnomAD_SAS", frequency=0.0, allele_count=0, allele_number=30604),
        ],
        pathogenicity_score=0.9997,
        pathogenicity_label="Pathogenic",
        prediction_confidence=0.998,
        feature_importances={
            "is_lof": 2.34,
            "cons_frameshift": 1.89,
            "gene_path_ratio": 1.52,
            "cons_synonymous": -1.41,
            "has_protein_change": 0.87,
        },
    )

    # ── Show evidence assembly ──
    print("=" * 70)
    print("  GenomicsGPT — Clinical Variant Report Generator")
    print("=" * 70)
    print(f"\n  Variant: {variant.display_name}")
    print(f"  Gene: {variant.gene_symbol}")
    print(f"  rsID: {variant.rs_id}")
    print(f"  Consequence: {variant.transcript.consequence.value}")
    print(f"  ML Score: {report.pathogenicity_score}")
    print(f"  ClinVar Records: {len(report.clinvar_records)}")

    evidence = assemble_evidence(report)
    prompt = build_prompt(evidence)

    print(f"\n{'─' * 70}")
    print("  Evidence prompt ({} chars):".format(len(prompt)))
    print(f"{'─' * 70}")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)

    # ── Generate report ──
    # Try Ollama first (free), fall back to Claude
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    generator = None

    try:
        from genomicsgpt.llm_engine.report_generator import OllamaReportGenerator
        print(f"\n{'─' * 70}")
        print("  Generating clinical narrative via Ollama (local, free)...")
        print(f"{'─' * 70}")
        generator = OllamaReportGenerator(model="llama3:latest")
    except ConnectionError:
        if api_key:
            print(f"\n{'─' * 70}")
            print("  Ollama not available, using Claude API...")
            print(f"{'─' * 70}")
            generator = ReportGenerator(api_key=api_key)
        else:
            print(f"\n{'─' * 70}")
            print("  ⚠ Ollama not running and no ANTHROPIC_API_KEY set.")
            print("  Option 1 (free): ollama serve  (in another terminal)")
            print("  Option 2 (paid): $env:ANTHROPIC_API_KEY='sk-ant-...'")
            print(f"{'─' * 70}")
            os.makedirs("data", exist_ok=True)
            with open("data/demo_prompt.txt", "w") as f:
                from genomicsgpt.llm_engine.report_generator import SYSTEM_PROMPT
                f.write(f"SYSTEM PROMPT:\n{'='*50}\n{SYSTEM_PROMPT}")
                f.write(f"\n\nUSER PROMPT:\n{'='*50}\n{prompt}")
            print("  Prompt saved to data/demo_prompt.txt")
            return None

    start = time.time()
    narrative = generator.generate(report)
    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print("  CLINICAL VARIANT REPORT")
    print(f"{'=' * 70}\n")
    print(narrative.full_report)
    print(f"\n{'─' * 70}")
    print(f"  Model: {narrative.model_used}")
    print(f"  Tokens: {narrative.tokens_used}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"{'─' * 70}")

    # Save report
    os.makedirs("data/reports", exist_ok=True)
    report_path = f"data/reports/{variant.gene_symbol}_{variant.transcript.hgvs_c.replace('.', '_')}.md"
    with open(report_path, "w") as f:
        f.write(f"# Clinical Variant Report: {variant.display_name}\n\n")
        f.write(f"*Generated by GenomicsGPT on {time.strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(narrative.full_report)
        f.write(f"\n\n---\n*Model: {narrative.model_used} | Tokens: {narrative.tokens_used} | Time: {elapsed:.2f}s*\n")
    print(f"\n  Report saved to {report_path}")

    return narrative


def demo_full_pipeline(variant_string: str):
    """
    Full end-to-end demo: parse → aggregate (live APIs) → ML predict → LLM narrative.
    """
    from genomicsgpt.llm_engine import generate_report

    print("=" * 70)
    print("  GenomicsGPT — Full Pipeline Demo")
    print("=" * 70)
    print(f"\n  Input: {variant_string}")
    print("  Running: parse → ClinVar → Ensembl VEP → ML → Claude\n")

    narrative = generate_report(variant_string)

    print(f"\n{'=' * 70}")
    print("  CLINICAL VARIANT REPORT")
    print(f"{'=' * 70}\n")
    print(narrative.full_report)

    return narrative


if __name__ == "__main__":
    variant = sys.argv[1] if len(sys.argv) > 1 else "BRCA1 c.5266dupC"

    # Use mock data demo (doesn't require ClinVar/Ensembl API calls)
    demo_with_mock_data(variant)
