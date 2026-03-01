"""GenomicsGPT CLI — Command-line interface for variant interpretation."""

from __future__ import annotations

import json
import sys


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m genomicsgpt <command> [args]")
        print("")
        print("Commands:")
        print("  parse <variant>     Parse a variant identifier")
        print("  clinvar <variant>   Query ClinVar for a variant")
        print("  interpret <variant> Full interpretation pipeline")
        print("")
        print("Examples:")
        print('  python -m genomicsgpt parse "BRCA1 c.5266dupC"')
        print('  python -m genomicsgpt parse "rs80357906"')
        print('  python -m genomicsgpt parse "chr17:43057051:A>T"')
        print('  python -m genomicsgpt clinvar "rs80357906"')
        return

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    if command == "parse":
        cmd_parse(args)
    elif command == "clinvar":
        cmd_clinvar(args)
    elif command == "interpret":
        cmd_interpret(args)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


def cmd_parse(args: list[str]):
    """Parse a variant and display results."""
    from genomicsgpt.variant_parser.parser import parse_variant

    if not args:
        print("Usage: python -m genomicsgpt parse <variant>")
        return

    raw = " ".join(args)
    v = parse_variant(raw)

    print(f"\n{'='*60}")
    print(f"  GenomicsGPT Variant Parser")
    print(f"{'='*60}")
    print(f"  Input:    {v.raw_input}")
    print(f"  Valid:    {v.is_valid}")
    print(f"  Display:  {v.display_name}")
    print(f"  Type:     {v.variant_type.value}")

    if v.rs_id:
        print(f"  rsID:     {v.rs_id}")
    if v.clinvar_id:
        print(f"  ClinVar:  {v.clinvar_id}")
    if v.gene_symbol:
        print(f"  Gene:     {v.gene_symbol}")

    if v.genomic:
        g = v.genomic
        print(f"  Genomic:  chr{g.chrom_normalized}:{g.position} {g.ref}>{g.alt} ({g.assembly.value})")

    if v.transcript:
        t = v.transcript
        if t.transcript_id:
            print(f"  Transcript: {t.transcript_id}")
        if t.hgvs_c:
            print(f"  HGVS c.:  {t.hgvs_c}")
        if t.hgvs_p:
            print(f"  HGVS p.:  {t.hgvs_p}")
        print(f"  Consequence: {t.consequence.value}")

    if v.parse_warnings:
        print(f"\n  Warnings:")
        for w in v.parse_warnings:
            print(f"    ⚠ {w}")

    print(f"{'='*60}\n")


def cmd_clinvar(args: list[str]):
    """Query ClinVar for a variant."""
    from genomicsgpt.variant_parser.parser import parse_variant
    from genomicsgpt.data_aggregator.clinvar_client import ClinVarClient

    if not args:
        print("Usage: python -m genomicsgpt clinvar <variant>")
        return

    raw = " ".join(args)
    v = parse_variant(raw)

    if not v.is_valid:
        print(f"Could not parse variant: {raw}")
        return

    print(f"\nQuerying ClinVar for: {v.display_name}...")
    client = ClinVarClient()
    records = client.query_variant(v)

    if not records:
        print("No ClinVar records found.")
        return

    print(f"\nFound {len(records)} ClinVar record(s):\n")
    for r in records:
        stars = "★" * r.review_stars + "☆" * (4 - r.review_stars)
        print(f"  [{stars}] {r.clinical_significance.value}")
        if r.condition:
            print(f"         Condition: {r.condition}")
        if r.review_status:
            print(f"         Review: {r.review_status}")
        if r.accession:
            print(f"         Accession: {r.accession}")
        print()


def cmd_interpret(args: list[str]):
    """Full interpretation pipeline."""
    from genomicsgpt.variant_parser.parser import parse_variant
    from genomicsgpt.data_aggregator.aggregator import DataAggregator

    if not args:
        print("Usage: python -m genomicsgpt interpret <variant>")
        return

    raw = " ".join(args)
    v = parse_variant(raw)

    if not v.is_valid:
        print(f"Could not parse variant: {raw}")
        return

    print(f"\n{'='*60}")
    print(f"  GenomicsGPT Variant Interpretation")
    print(f"{'='*60}")
    print(f"  Input: {v.raw_input}")
    print(f"  Parsed as: {v.display_name}")
    print(f"\n  Aggregating data from multiple sources...")
    print(f"  [1/2] Querying ClinVar...", end=" ", flush=True)

    agg = DataAggregator()
    report = agg.aggregate(v)
    
    print(f"done.")

    # ── Variant Identity ──
    print(f"\n  --- Variant Identity ---")
    if report.variant.gene_symbol:
        print(f"  Gene:       {report.variant.gene_symbol}")
    if report.variant.rs_id:
        print(f"  rsID:       {report.variant.rs_id}")
    if report.variant.transcript:
        t = report.variant.transcript
        if t.transcript_id:
            print(f"  Transcript: {t.transcript_id}")
        if t.hgvs_c:
            print(f"  HGVS c.:   {t.hgvs_c}")
        if t.hgvs_p:
            print(f"  HGVS p.:   {t.hgvs_p}")
        print(f"  Consequence: {t.consequence.value}")
    print(f"  Type:       {report.variant.variant_type.value}")

    # ── ClinVar ──
    if report.clinvar_records:
        print(f"\n  --- ClinVar ({len(report.clinvar_records)} records) ---")
        for r in report.clinvar_records:
            stars = "★" * r.review_stars + "☆" * (4 - r.review_stars)
            print(f"  [{stars}] {r.clinical_significance.value}")
            if r.condition:
                print(f"           {r.condition}")

    # ── Population Frequencies ──
    if report.population_frequencies:
        print(f"\n  --- Population Frequencies ---")
        # Show top 5 by frequency
        sorted_freqs = sorted(report.population_frequencies, key=lambda x: x.frequency, reverse=True)
        for pf in sorted_freqs[:5]:
            bar = "█" * int(pf.frequency * 500) if pf.frequency > 0 else "░"
            print(f"  {pf.population:20s}  {pf.frequency:.6f}  {bar}")
        max_f = report.max_gnomad_frequency
        if max_f is not None:
            print(f"  Max frequency:        {max_f:.6f}")

    # ── Protein Domains ──
    if report.protein_domains:
        print(f"\n  --- Protein Domains ---")
        seen = set()
        for d in report.protein_domains:
            key = f"{d.name}:{d.description}"
            if key not in seen:
                print(f"  {d.source}: {d.description} ({d.start}-{d.end})")
                seen.add(key)

    # ── Scores ──
    scores = report.feature_importances
    if scores:
        print(f"\n  --- Prediction Scores ---")
        if "cadd_phred" in scores:
            cadd = scores["cadd_phred"]
            label = "HIGH" if cadd > 20 else "MODERATE" if cadd > 15 else "LOW"
            print(f"  CADD Phred: {cadd:.1f} ({label})")
        if "most_severe_consequence" in scores:
            print(f"  VEP consequence: {scores['most_severe_consequence']}")

    print(f"\n  --- Pipeline ---")
    print(f"  Processing time: {report.processing_time_seconds:.2f}s")
    print(f"  Version: {report.pipeline_version}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
