"""
ClinVar API Client — Fetches variant annotations from NCBI ClinVar.

Uses NCBI E-utilities (esearch + esummary) to query ClinVar programmatically.
Implements caching to avoid redundant API calls.

API docs: https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from genomicsgpt.models import (
    ClinicalSignificance,
    ClinVarRecord,
    ParsedVariant,
    PopulationFrequency,
)

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CLINVAR_DB = "clinvar"
RATE_LIMIT_DELAY = 0.34  # NCBI allows 3 requests/sec without API key


def _map_significance(raw: str) -> ClinicalSignificance:
    """Map ClinVar significance string to enum."""
    low = raw.lower().strip()
    mapping = {
        "pathogenic": ClinicalSignificance.PATHOGENIC,
        "likely pathogenic": ClinicalSignificance.LIKELY_PATHOGENIC,
        "uncertain significance": ClinicalSignificance.UNCERTAIN,
        "likely benign": ClinicalSignificance.LIKELY_BENIGN,
        "benign": ClinicalSignificance.BENIGN,
        "conflicting classifications of pathogenicity": ClinicalSignificance.CONFLICTING,
        "conflicting interpretations of pathogenicity": ClinicalSignificance.CONFLICTING,
    }
    for key, val in mapping.items():
        if key in low:
            return val
    return ClinicalSignificance.NOT_PROVIDED


def _map_review_stars(status: str) -> int:
    """Map review status to star rating (0-4)."""
    status_low = status.lower()
    if "expert panel" in status_low:
        return 3
    if "practice guideline" in status_low:
        return 4
    if "multiple submitters" in status_low and "no conflicts" in status_low:
        return 2
    if "single submitter" in status_low or "criteria provided" in status_low:
        return 1
    return 0


class ClinVarClient:
    """Client for querying ClinVar via NCBI E-utilities."""

    def __init__(
        self,
        email: str = "genomicsgpt@example.com",
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.email = email
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})
        self._last_request_time = 0.0
        
        # Simple file-based cache
        self.cache_dir = cache_dir or Path.home() / ".genomicsgpt" / "cache" / "clinvar"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _rate_limit(self):
        """Respect NCBI rate limits."""
        elapsed = time.time() - self._last_request_time
        delay = 0.11 if self.api_key else RATE_LIMIT_DELAY
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_request_time = time.time()

    def _base_params(self) -> dict:
        """Base parameters for all E-utility requests."""
        params = {"email": self.email, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _get_cached(self, cache_key: str) -> Optional[dict]:
        """Check file cache."""
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return None

    def _set_cache(self, cache_key: str, data: dict):
        """Write to file cache."""
        try:
            path = self.cache_dir / f"{cache_key}.json"
            path.write_text(json.dumps(data))
        except OSError:
            pass

    def search(self, query: str, max_results: int = 20) -> list[int]:
        """
        Search ClinVar and return variation IDs.
        
        Args:
            query: ClinVar search query (e.g., "BRCA1[gene] AND pathogenic")
            max_results: Maximum number of results
            
        Returns:
            List of ClinVar variation IDs
        """
        self._rate_limit()
        params = {
            **self._base_params(),
            "db": CLINVAR_DB,
            "term": query,
            "retmax": max_results,
        }
        
        try:
            resp = self.session.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            return [int(x) for x in id_list]
        except requests.exceptions.Timeout:
            print("  ⚠ ClinVar search timed out (10s). Check your internet connection.")
            return []
        except requests.exceptions.ConnectionError:
            print("  ⚠ Could not connect to ClinVar. Check your internet/firewall.")
            return []
        except Exception as e:
            logger.error(f"ClinVar search failed: {e}")
            return []

    def fetch_summary(self, variation_ids: list[int]) -> list[dict]:
        """
        Fetch summary data for ClinVar variation IDs.
        
        Args:
            variation_ids: List of ClinVar variation IDs
            
        Returns:
            List of raw summary dictionaries
        """
        if not variation_ids:
            return []

        # Check cache
        cache_key = f"summary_{'_'.join(str(v) for v in sorted(variation_ids[:5]))}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit()
        params = {
            **self._base_params(),
            "db": CLINVAR_DB,
            "id": ",".join(str(v) for v in variation_ids),
        }
        
        try:
            resp = self.session.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("result", {})
            
            summaries = []
            for vid in variation_ids:
                entry = results.get(str(vid))
                if entry:
                    summaries.append(entry)
            
            self._set_cache(cache_key, summaries)
            return summaries
        except requests.exceptions.Timeout:
            print("  ⚠ ClinVar fetch timed out (10s).")
            return []
        except requests.exceptions.ConnectionError:
            print("  ⚠ Could not connect to ClinVar.")
            return []
        except Exception as e:
            logger.error(f"ClinVar fetch failed: {e}")
            return []

    def query_variant(self, variant: ParsedVariant) -> list[ClinVarRecord]:
        """
        Query ClinVar for a parsed variant using the best available identifier.
        
        Args:
            variant: Parsed variant object
            
        Returns:
            List of ClinVar records for this variant
        """
        # Build query based on what identifiers we have
        query = self._build_query(variant)
        if not query:
            logger.warning(f"Cannot build ClinVar query for: {variant.raw_input}")
            return []
        
        logger.info(f"ClinVar query: {query}")
        ids = self.search(query)
        if not ids:
            return []
        
        summaries = self.fetch_summary(ids)
        return self._parse_summaries(summaries)

    def _build_query(self, variant: ParsedVariant) -> Optional[str]:
        """Build a ClinVar search query from parsed variant."""
        # Prefer ClinVar ID
        if variant.clinvar_id:
            return f"{variant.clinvar_id}[uid]"
        
        # Try rsID
        if variant.rs_id:
            return f"{variant.rs_id}[dbSNP ID]"
        
        # Try gene + HGVS
        if variant.transcript:
            t = variant.transcript
            if t.transcript_id and t.hgvs_c:
                return f"{t.transcript_id}:{t.hgvs_c}"
            if t.gene_symbol and t.hgvs_c:
                return f"{t.gene_symbol}[gene] AND {t.hgvs_c}"
            if t.gene_symbol and t.hgvs_p:
                return f"{t.gene_symbol}[gene] AND {t.hgvs_p}"
        
        # Try gene alone
        if variant.gene_symbol:
            return f"{variant.gene_symbol}[gene]"
        
        # Try coordinates
        if variant.genomic:
            g = variant.genomic
            return f"chr{g.chrom_normalized}[chr] AND {g.position}[chrpos]"
        
        return None

    def _parse_summaries(self, summaries: list[dict]) -> list[ClinVarRecord]:
        """Parse raw ClinVar summaries into ClinVarRecord objects."""
        records = []
        for s in summaries:
            try:
                # ClinVar API uses germline_classification (newer) or clinical_significance
                germ = s.get("germline_classification", {})
                if isinstance(germ, dict) and germ.get("description"):
                    sig_raw = germ.get("description", "")
                    review = germ.get("review_status", "")
                    last_eval = germ.get("last_evaluated", "")
                    trait_set = germ.get("trait_set", [])
                else:
                    # Fallback to older field
                    cs = s.get("clinical_significance", {})
                    if isinstance(cs, dict):
                        sig_raw = cs.get("description", "")
                        review = cs.get("review_status", "")
                        last_eval = cs.get("last_evaluated", "")
                        trait_set = cs.get("trait_set", [])
                    else:
                        sig_raw = str(cs)
                        review = ""
                        last_eval = ""
                        trait_set = []

                # Extract conditions from trait_set
                conditions = []
                if isinstance(trait_set, list):
                    for trait in trait_set:
                        if isinstance(trait, dict):
                            name = trait.get("trait_name", "")
                            if name:
                                conditions.append(name)
                
                record = ClinVarRecord(
                    variation_id=int(s.get("uid", 0)),
                    clinical_significance=_map_significance(str(sig_raw)),
                    review_status=str(review),
                    review_stars=_map_review_stars(str(review)),
                    condition="; ".join(conditions) if conditions else "",
                    accession=s.get("accession", ""),
                    last_evaluated=str(last_eval) if last_eval else None,
                )
                records.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse ClinVar summary: {e}")
                continue
        
        return records

    def query_by_rsid(self, rs_id: str) -> list[ClinVarRecord]:
        """Convenience: query ClinVar by rsID."""
        v = ParsedVariant(rs_id=rs_id, raw_input=rs_id)
        return self.query_variant(v)

    def query_by_gene(
        self, gene: str, significance: Optional[str] = None, max_results: int = 100
    ) -> list[ClinVarRecord]:
        """Query all ClinVar variants for a gene, optionally filtered by significance."""
        query = f"{gene}[gene]"
        if significance:
            query += f" AND {significance}[clinical significance]"
        ids = self.search(query, max_results=max_results)
        if not ids:
            return []
        # Batch in groups of 50
        records = []
        for i in range(0, len(ids), 50):
            batch = ids[i : i + 50]
            summaries = self.fetch_summary(batch)
            records.extend(self._parse_summaries(summaries))
        return records
