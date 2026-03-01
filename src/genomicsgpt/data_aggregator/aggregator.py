"""
Data Aggregator — Orchestrates multi-source data retrieval for a variant.

Queries ClinVar, Ensembl VEP, and (future) gnomAD/AlphaMissense/UniProt,
then assembles everything into a unified VariantReport.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from genomicsgpt.models import (
    AlphaMissenseScore,
    ParsedVariant,
    PopulationFrequency,
    VariantReport,
)
from genomicsgpt.data_aggregator.clinvar_client import ClinVarClient
from genomicsgpt.data_aggregator.ensembl_client import EnsemblClient

logger = logging.getLogger(__name__)


class DataAggregator:
    """
    Orchestrates data retrieval from multiple genomic databases.

    Usage:
        agg = DataAggregator()
        report = agg.aggregate(parsed_variant)
    """

    def __init__(
        self,
        clinvar_client: Optional[ClinVarClient] = None,
        ensembl_client: Optional[EnsemblClient] = None,
    ):
        self.clinvar = clinvar_client or ClinVarClient()
        self.ensembl = ensembl_client or EnsemblClient()

    def aggregate(self, variant: ParsedVariant) -> VariantReport:
        """
        Aggregate data from all sources for a variant.

        Args:
            variant: Parsed variant with at least one identifier.

        Returns:
            VariantReport with all available annotations.
        """
        start_time = time.time()
        report = VariantReport(variant=variant)

        if not variant.is_valid:
            logger.error(f"Invalid variant: {variant.raw_input}")
            return report

        # ── Step 1: ClinVar ──
        logger.info("Querying ClinVar...")
        try:
            report.clinvar_records = self.clinvar.query_variant(variant)
            logger.info(f"  Found {len(report.clinvar_records)} ClinVar records")
        except Exception as e:
            logger.error(f"  ClinVar query failed: {e}")

        # ── Step 2: Ensembl VEP ──
        print("  [2/2] Querying Ensembl VEP...", end=" ", flush=True)
        logger.info("Querying Ensembl VEP...")
        try:
            vep_data = self.ensembl.annotate_variant(variant)
            if vep_data:
                # Population frequencies
                report.population_frequencies = vep_data.get("frequencies", [])
                logger.info(f"  Found {len(report.population_frequencies)} frequency entries")

                # Protein domains
                report.protein_domains = vep_data.get("protein_domains", [])
                logger.info(f"  Found {len(report.protein_domains)} protein domains")

                # Enrich variant info from VEP
                self._enrich_variant_from_vep(variant, vep_data)

                # Store CADD for later ML features
                cadd = vep_data.get("cadd_score")
                if cadd is not None:
                    report.feature_importances["cadd_phred"] = float(cadd)
                    logger.info(f"  CADD phred: {cadd}")

                # Store consequence
                msq = vep_data.get("most_severe_consequence", "")
                if msq:
                    report.feature_importances["most_severe_consequence"] = msq
                    logger.info(f"  Most severe consequence: {msq}")

        except Exception as e:
            logger.error(f"  Ensembl VEP query failed: {e}")

        # ── Step 3: AlphaMissense (placeholder - will use DuckDB index) ──
        # TODO: Implement in Week 1 continued
        # report.alphamissense = self._query_alphamissense(variant)

        report.processing_time_seconds = time.time() - start_time
        logger.info(f"Aggregation complete in {report.processing_time_seconds:.2f}s")
        return report

    def _enrich_variant_from_vep(self, variant: ParsedVariant, vep_data: dict):
        """Update the ParsedVariant with information from VEP."""
        consequences = vep_data.get("transcript_consequences", [])

        # Find canonical transcript consequence
        canonical = None
        for tc in consequences:
            if tc.get("canonical"):
                canonical = tc
                break
        if not canonical and consequences:
            canonical = consequences[0]

        if canonical:
            # Fill in gene symbol if missing
            if not variant.gene_symbol:
                variant.gene_symbol = canonical.get("gene_symbol", "")

            # Fill in transcript info if missing
            if variant.transcript:
                if not variant.transcript.gene_symbol:
                    variant.transcript.gene_symbol = canonical.get("gene_symbol", "")
                if not variant.transcript.hgvs_p and canonical.get("hgvsp"):
                    variant.transcript.hgvs_p = canonical["hgvsp"].split(":")[-1]
            elif canonical.get("gene_symbol"):
                from genomicsgpt.models import TranscriptVariant, MolecularConsequence
                variant.transcript = TranscriptVariant(
                    transcript_id=canonical.get("transcript_id", ""),
                    gene_symbol=canonical.get("gene_symbol", ""),
                    hgvs_c=canonical.get("hgvsc", "").split(":")[-1] if canonical.get("hgvsc") else "",
                    hgvs_p=canonical.get("hgvsp", "").split(":")[-1] if canonical.get("hgvsp") else "",
                )


def quick_lookup(variant_string: str) -> VariantReport:
    """
    One-liner convenience function for quick variant lookup.

    Args:
        variant_string: Any variant format (rsID, HGVS, coords, gene+mutation)

    Returns:
        VariantReport with all available data.
    """
    from genomicsgpt.variant_parser.parser import parse_variant

    parsed = parse_variant(variant_string)
    aggregator = DataAggregator()
    return aggregator.aggregate(parsed)
