"""
Variant Parser — Parses genetic variant identifiers from multiple input formats.

Supported formats:
  - HGVS transcript:  NM_007294.4:c.5266dupC
  - HGVS protein:     NP_009225.1:p.Gln1756Profs*74
  - rsID:             rs80357906
  - Genomic coords:   chr17:43057051:A>T  or  17-43057051-A-T
  - Gene + mutation:  BRCA1 c.5266dupC  or  BRCA1 V600E
  - ClinVar ID:       ClinVar:12345  or  VCV000012345
"""

from __future__ import annotations

import re
from typing import Optional

from genomicsgpt.models import (
    Assembly,
    GenomicPosition,
    MolecularConsequence,
    ParsedVariant,
    TranscriptVariant,
    VariantType,
)

# ═══════════════════════════════════════════════════════════════
# Regex patterns for variant formats
# ═══════════════════════════════════════════════════════════════

# HGVS transcript: NM_007294.4:c.5266dupC
RE_HGVS_C = re.compile(
    r"(?P<transcript>N[MR]_\d+(?:\.\d+)?)"
    r":(?P<change>c\.\S+)"
)

# HGVS protein: NP_009225.1:p.Gln1756Profs*74 or p.V600E
RE_HGVS_P = re.compile(
    r"(?P<transcript>NP_\d+(?:\.\d+)?)"
    r":(?P<change>p\.\S+)"
)

# rsID: rs80357906
RE_RSID = re.compile(r"^(?:rs)(?P<id>\d+)$", re.IGNORECASE)

# Genomic coordinates: chr17:43057051:A>T or 17:43057051:A:T or 17-43057051-A-T
RE_COORD = re.compile(
    r"^(?:chr)?(?P<chrom>[\dXYM]+[T]?)"
    r"[:\-_](?P<pos>\d+)"
    r"[:\-_](?P<ref>[ACGTNacgtn]+)"
    r"[>:\-_](?P<alt>[ACGTNacgtn]+)$"
)

# Gene + cDNA: BRCA1 c.5266dupC
RE_GENE_CDNA = re.compile(
    r"^(?P<gene>[A-Z][A-Z0-9\-]+)"
    r"\s+(?P<change>c\.\S+)$",
    re.IGNORECASE,
)

# Gene + protein change: BRCA1 V600E or BRCA1 p.Val600Glu
RE_GENE_PROTEIN = re.compile(
    r"^(?P<gene>[A-Z][A-Z0-9\-]+)"
    r"\s+(?:p\.)?(?P<change>[A-Z][a-z]{2}\d+[A-Z][a-z]{2}\S*|[A-Z]\d+[A-Z*])$",
    re.IGNORECASE,
)

# ClinVar accession: VCV000012345 or ClinVar:12345
RE_CLINVAR = re.compile(
    r"^(?:VCV0*|clinvar[:\s]*)(?P<id>\d+)$", re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════
# Amino acid code mappings
# ═══════════════════════════════════════════════════════════════

AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Glu": "E", "Gln": "Q", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Ter": "*", "Sec": "U",
}

AA_1TO3 = {v: k for k, v in AA_3TO1.items()}


def infer_variant_type(hgvs_notation: str) -> VariantType:
    """Infer variant type from HGVS notation."""
    h = hgvs_notation.lower()
    if "dup" in h:
        return VariantType.DUPLICATION
    if "del" in h and "ins" in h:
        return VariantType.INDEL
    if "del" in h:
        return VariantType.DELETION
    if "ins" in h:
        return VariantType.INSERTION
    if ">" in h:
        return VariantType.SNV
    if "fs" in h:
        return VariantType.FRAMESHIFT
    return VariantType.UNKNOWN


def infer_consequence(hgvs_c: Optional[str], hgvs_p: Optional[str]) -> MolecularConsequence:
    """Infer molecular consequence from HGVS notation."""
    if hgvs_p:
        p = hgvs_p
        if "fs" in p or "Ter" in p and "ext" not in p:
            if "fs" in p:
                return MolecularConsequence.FRAMESHIFT
            return MolecularConsequence.NONSENSE
        if p.startswith("p.Met1"):
            return MolecularConsequence.START_LOST
        if "=" in p:
            return MolecularConsequence.SYNONYMOUS
        return MolecularConsequence.MISSENSE

    if hgvs_c:
        c = hgvs_c.lower()
        # Splice variants
        if any(x in c for x in ["+1", "+2", "-1", "-2"]) and any(
            x in c for x in ["a>", "g>", "c>", "t>", "del", "dup"]
        ):
            if "+1" in c or "+2" in c:
                return MolecularConsequence.SPLICE_DONOR
            return MolecularConsequence.SPLICE_ACCEPTOR
        if "del" in c and "ins" in c:
            return MolecularConsequence.MISSENSE  # could be inframe or frameshift
        if "dup" in c or "ins" in c:
            return MolecularConsequence.FRAMESHIFT
        if "del" in c:
            return MolecularConsequence.FRAMESHIFT
        if ">" in c:
            return MolecularConsequence.MISSENSE  # SNV — need protein to confirm

    return MolecularConsequence.UNKNOWN


def parse_variant(raw_input: str) -> ParsedVariant:
    """
    Parse a variant from any supported input format.
    
    Args:
        raw_input: Variant string in any format (HGVS, rsID, coordinates, etc.)
        
    Returns:
        ParsedVariant with all identifiers that could be extracted.
        
    Examples:
        >>> parse_variant("rs80357906")
        ParsedVariant(rs_id='rs80357906', ...)
        
        >>> parse_variant("NM_007294.4:c.5266dupC")
        ParsedVariant(transcript=TranscriptVariant(...), ...)
        
        >>> parse_variant("chr17:43057051:A>T")
        ParsedVariant(genomic=GenomicPosition(...), ...)
        
        >>> parse_variant("BRCA1 c.5266dupC")
        ParsedVariant(gene_symbol='BRCA1', transcript=TranscriptVariant(...), ...)
    """
    text = raw_input.strip()
    result = ParsedVariant(raw_input=text)
    
    if not text:
        result.parse_warnings.append("Empty input")
        return result

    # ── Try rsID ──
    m = RE_RSID.match(text)
    if m:
        result.rs_id = f"rs{m.group('id')}"
        return result

    # ── Try ClinVar ID ──
    m = RE_CLINVAR.match(text)
    if m:
        result.clinvar_id = int(m.group("id"))
        return result

    # ── Try genomic coordinates ──
    m = RE_COORD.match(text)
    if m:
        result.genomic = GenomicPosition(
            chromosome=m.group("chrom"),
            position=int(m.group("pos")),
            ref=m.group("ref").upper(),
            alt=m.group("alt").upper(),
        )
        result.variant_type = _infer_type_from_alleles(
            result.genomic.ref, result.genomic.alt
        )
        return result

    # ── Try HGVS transcript ──
    m = RE_HGVS_C.search(text)
    if m:
        hgvs_c = m.group("change")
        result.transcript = TranscriptVariant(
            transcript_id=m.group("transcript"),
            gene_symbol="",
            hgvs_c=hgvs_c,
            consequence=infer_consequence(hgvs_c, None),
        )
        result.variant_type = infer_variant_type(hgvs_c)
        # Check if gene symbol is also in the input
        gene_part = text[: m.start()].strip().rstrip(":")
        if gene_part and re.match(r"^[A-Z][A-Z0-9\-]+$", gene_part, re.IGNORECASE):
            result.gene_symbol = gene_part.upper()
            result.transcript.gene_symbol = result.gene_symbol
        return result

    # ── Try HGVS protein ──
    m = RE_HGVS_P.search(text)
    if m:
        hgvs_p = m.group("change")
        result.transcript = TranscriptVariant(
            transcript_id=m.group("transcript"),
            gene_symbol="",
            hgvs_c="",
            hgvs_p=hgvs_p,
            consequence=infer_consequence(None, hgvs_p),
        )
        return result

    # ── Try gene + cDNA ──
    m = RE_GENE_CDNA.match(text)
    if m:
        gene = m.group("gene").upper()
        hgvs_c = m.group("change")
        result.gene_symbol = gene
        result.transcript = TranscriptVariant(
            transcript_id="",  # Will be resolved by data aggregator
            gene_symbol=gene,
            hgvs_c=hgvs_c,
            consequence=infer_consequence(hgvs_c, None),
        )
        result.variant_type = infer_variant_type(hgvs_c)
        result.parse_warnings.append(
            f"Transcript ID not provided for {gene}; will attempt auto-resolution"
        )
        return result

    # ── Try gene + protein change ──
    m = RE_GENE_PROTEIN.match(text)
    if m:
        gene = m.group("gene").upper()
        change = m.group("change")
        # Normalize short form (V600E → p.Val600Glu)
        hgvs_p = _normalize_protein_change(change)
        result.gene_symbol = gene
        result.transcript = TranscriptVariant(
            transcript_id="",
            gene_symbol=gene,
            hgvs_c="",
            hgvs_p=hgvs_p,
            consequence=MolecularConsequence.MISSENSE,
        )
        result.variant_type = VariantType.SNV
        result.parse_warnings.append(
            f"Transcript ID not provided for {gene}; will attempt auto-resolution"
        )
        return result

    # ── Nothing matched ──
    result.parse_warnings.append(
        f"Could not parse variant format: '{text}'. "
        "Supported formats: HGVS (NM_:c.), rsID, coordinates (chr:pos:ref>alt), "
        "gene+mutation (BRCA1 c.5266dupC or BRAF V600E)"
    )
    return result


def _infer_type_from_alleles(ref: str, alt: str) -> VariantType:
    """Infer variant type from ref/alt alleles."""
    if len(ref) == 1 and len(alt) == 1:
        return VariantType.SNV
    if len(ref) > len(alt):
        return VariantType.DELETION
    if len(ref) < len(alt):
        return VariantType.INSERTION
    if len(ref) > 1:
        return VariantType.MNV
    return VariantType.UNKNOWN


def _normalize_protein_change(change: str) -> str:
    """Normalize protein change to HGVS p. notation.
    
    V600E → p.Val600Glu
    p.Val600Glu → p.Val600Glu (unchanged)
    """
    if change.startswith("p."):
        return change
    
    # Short form: V600E
    m = re.match(r"([A-Z])(\d+)([A-Z*])", change)
    if m:
        ref_aa = AA_1TO3.get(m.group(1), m.group(1))
        pos = m.group(2)
        alt_aa = AA_1TO3.get(m.group(3), m.group(3))
        return f"p.{ref_aa}{pos}{alt_aa}"
    
    return f"p.{change}"


def parse_variants_batch(inputs: list[str]) -> list[ParsedVariant]:
    """Parse multiple variant strings."""
    return [parse_variant(v) for v in inputs]
