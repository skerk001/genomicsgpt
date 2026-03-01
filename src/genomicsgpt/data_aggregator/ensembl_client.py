"""
Ensembl REST API Client — Variant Effect Predictor (VEP) and gene lookups.

Provides:
  - VEP annotation (consequence, SIFT, PolyPhen, CADD)
  - Gene/transcript resolution (symbol → canonical transcript)
  - Coordinate mapping and liftover
  
API docs: https://rest.ensembl.org/
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from genomicsgpt.models import (
    GenomicPosition,
    MolecularConsequence,
    ParsedVariant,
    PopulationFrequency,
    ProteinDomain,
    TranscriptVariant,
)

logger = logging.getLogger(__name__)

ENSEMBL_REST = "https://rest.ensembl.org"
RATE_LIMIT_DELAY = 0.07  # Ensembl allows 15 req/sec


class EnsemblClient:
    """Client for Ensembl REST API."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        self._last_request = 0.0

    def _get(self, endpoint: str, params: Optional[dict] = None) -> Optional[dict | list]:
        """Make rate-limited GET request to Ensembl REST."""
        elapsed = time.time() - self._last_request
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request = time.time()

        url = f"{ENSEMBL_REST}/{endpoint.lstrip('/')}"
        try:
            resp = self.session.get(url, params=params, timeout=10)
            if resp.status_code == 429:
                retry = float(resp.headers.get("Retry-After", 1))
                logger.warning(f"Rate limited, retrying in {retry}s")
                time.sleep(retry)
                return self._get(endpoint, params)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"  ⚠ Ensembl request timed out (10s).")
            return None
        except requests.exceptions.ConnectionError:
            print(f"  ⚠ Could not connect to Ensembl. Check your internet/firewall.")
            return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 400:
                logger.warning(f"Ensembl 400 for {endpoint}: {e}")
            else:
                logger.error(f"Ensembl request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Ensembl request error: {e}")
            return None

    def vep_hgvs(self, hgvs_notation: str) -> Optional[dict]:
        """
        Run VEP on an HGVS notation.

        Args:
            hgvs_notation: e.g., "NM_007294.4:c.5266dupC"

        Returns:
            VEP result dictionary with consequences, frequencies, predictions.
        """
        data = self._get(f"/vep/human/hgvs/{hgvs_notation}", params={
            "content-type": "application/json",
            "CADD": "1",
            "gnomADe": "1",
            "gnomADg": "1",
            "Conservation": "1",
            "protein": "1",
            "domains": "1",
            "hgvs": "1",
        })
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else None

    def vep_region(self, chrom: str, pos: int, ref: str, alt: str) -> Optional[dict]:
        """
        Run VEP on genomic coordinates.

        Args:
            chrom: Chromosome (e.g., "17")
            pos: Position
            ref: Reference allele
            alt: Alternate allele

        Returns:
            VEP result dictionary.
        """
        region = f"{chrom}:{pos}:{ref}/{alt}"
        data = self._get(f"/vep/human/region/{region}", params={
            "content-type": "application/json",
            "CADD": "1",
            "gnomADe": "1",
            "gnomADg": "1",
            "Conservation": "1",
            "protein": "1",
            "domains": "1",
            "hgvs": "1",
        })
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else None

    def vep_rsid(self, rs_id: str) -> Optional[dict]:
        """
        Run VEP on an rsID.

        Args:
            rs_id: e.g., "rs80357906"

        Returns:
            VEP result dictionary.
        """
        data = self._get(f"/vep/human/id/{rs_id}", params={
            "content-type": "application/json",
            "CADD": "1",
            "gnomADe": "1",
            "gnomADg": "1",
            "Conservation": "1",
            "protein": "1",
            "domains": "1",
            "hgvs": "1",
        })
        if isinstance(data, list) and data:
            return data[0]
        return data if isinstance(data, dict) else None

    def annotate_variant(self, variant: ParsedVariant) -> dict[str, Any]:
        """
        Run VEP annotation for a ParsedVariant using best available identifier.

        Returns dict with keys:
            - transcript_consequences: list of consequence annotations
            - genomic_position: resolved genomic coords
            - colocated_variants: known variants at this position
            - most_severe_consequence: top consequence string
        """
        vep_result = None

        # Priority: HGVS > rsID > coordinates > gene lookup
        if variant.transcript and variant.transcript.transcript_id and variant.transcript.hgvs_c:
            hgvs = f"{variant.transcript.transcript_id}:{variant.transcript.hgvs_c}"
            logger.info(f"VEP query by HGVS: {hgvs}")
            vep_result = self.vep_hgvs(hgvs)

        if not vep_result and variant.rs_id:
            logger.info(f"VEP query by rsID: {variant.rs_id}")
            vep_result = self.vep_rsid(variant.rs_id)

        if not vep_result and variant.genomic:
            g = variant.genomic
            logger.info(f"VEP query by region: {g}")
            vep_result = self.vep_region(g.chrom_normalized, g.position, g.ref, g.alt)

        # If we only have gene + protein change, try to resolve via gene lookup + VEP HGVS
        if not vep_result and variant.gene_symbol and variant.transcript:
            t = variant.transcript
            if t.hgvs_p and not t.hgvs_c:
                # Try Ensembl gene lookup to get canonical transcript, then VEP with protein HGVS
                gene_info = self.get_gene_info(variant.gene_symbol)
                if gene_info and gene_info.get("canonical_transcript"):
                    ensembl_tx = gene_info["canonical_transcript"]
                    # Try protein-level HGVS via the Ensembl protein ID
                    logger.info(f"Resolved {variant.gene_symbol} → {ensembl_tx}, trying VEP")
                    # For protein changes, we need to search ClinVar/Ensembl differently
                    # Try VEP with the gene symbol directly via /vep/human/hgvs/
                    # Ensembl VEP doesn't accept bare gene+protein, so we note this limitation
                    variant.parse_warnings.append(
                        f"Resolved canonical transcript: {ensembl_tx}. "
                        "For full VEP annotation, provide transcript-level HGVS (e.g., NM_004333.6:c.1799T>A)"
                    )

        if not vep_result:
            logger.warning(f"No VEP annotation available for: {variant.display_name}")
            return {}

        return self._extract_annotation(vep_result)

    def _extract_annotation(self, vep: dict) -> dict[str, Any]:
        """Extract structured annotation from raw VEP response."""
        result: dict[str, Any] = {
            "most_severe_consequence": vep.get("most_severe_consequence", ""),
            "transcript_consequences": [],
            "frequencies": [],
            "protein_domains": [],
            "cadd_score": None,
            "conservation_score": None,
        }

        # Transcript consequences
        for tc in vep.get("transcript_consequences", []):
            cons = {
                "gene_symbol": tc.get("gene_symbol", ""),
                "gene_id": tc.get("gene_id", ""),
                "transcript_id": tc.get("transcript_id", ""),
                "consequence_terms": tc.get("consequence_terms", []),
                "impact": tc.get("impact", ""),
                "biotype": tc.get("biotype", ""),
                "canonical": tc.get("canonical", 0) == 1,
                "hgvsc": tc.get("hgvsc", ""),
                "hgvsp": tc.get("hgvsp", ""),
                "protein_start": tc.get("protein_start"),
                "protein_end": tc.get("protein_end"),
                "amino_acids": tc.get("amino_acids", ""),
                "codons": tc.get("codons", ""),
                "sift_prediction": tc.get("sift_prediction", ""),
                "sift_score": tc.get("sift_score"),
                "polyphen_prediction": tc.get("polyphen_prediction", ""),
                "polyphen_score": tc.get("polyphen_score"),
                "cadd_phred": tc.get("cadd_phred"),
                "cadd_raw": tc.get("cadd_raw"),
            }
            result["transcript_consequences"].append(cons)

            # Extract CADD from canonical transcript
            if cons["canonical"] and cons.get("cadd_phred"):
                result["cadd_score"] = cons["cadd_phred"]

            # Extract protein domains
            for dom in tc.get("domains", []):
                result["protein_domains"].append(ProteinDomain(
                    name=dom.get("db", ""),
                    start=tc.get("protein_start") or 0,
                    end=tc.get("protein_end") or 0,
                    source=dom.get("db", ""),
                    description=dom.get("name", ""),
                ))

        # Colocated variants (gnomAD frequencies)
        for cv in vep.get("colocated_variants", []):
            freqs = cv.get("frequencies", {})
            for allele, pop_data in freqs.items():
                if isinstance(pop_data, dict):
                    for pop, freq in pop_data.items():
                        if isinstance(freq, (int, float)):
                            result["frequencies"].append(PopulationFrequency(
                                population=f"gnomAD_{pop}",
                                frequency=float(freq),
                            ))

        return result

    def get_gene_info(self, gene_symbol: str) -> Optional[dict]:
        """
        Look up gene info by symbol.

        Returns:
            Dict with gene_id, description, canonical transcript, etc.
        """
        data = self._get(f"/lookup/symbol/homo_sapiens/{gene_symbol}", params={
            "expand": "1",
        })
        if not data:
            return None

        info = {
            "gene_id": data.get("id", ""),
            "display_name": data.get("display_name", ""),
            "description": data.get("description", ""),
            "biotype": data.get("biotype", ""),
            "chromosome": data.get("seq_region_name", ""),
            "start": data.get("start"),
            "end": data.get("end"),
            "strand": data.get("strand"),
        }

        # Find canonical transcript
        for tx in data.get("Transcript", []):
            if tx.get("is_canonical"):
                info["canonical_transcript"] = tx.get("id", "")
                info["canonical_transcript_refseq"] = ""
                # Try to get RefSeq ID
                for xref in tx.get("Translation", {}).get("db_links", []) if tx.get("Translation") else []:
                    if xref.get("dbname") == "RefSeq_mRNA":
                        info["canonical_transcript_refseq"] = xref.get("primary_id", "")
                break

        return info

    def resolve_transcript(self, gene_symbol: str) -> Optional[str]:
        """Get the canonical Ensembl transcript ID for a gene symbol."""
        info = self.get_gene_info(gene_symbol)
        if info:
            return info.get("canonical_transcript")
        return None
