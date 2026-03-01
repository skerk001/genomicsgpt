"""Core data models for GenomicsGPT variant interpretation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Assembly(str, Enum):
    """Human genome assembly versions."""
    GRCH37 = "GRCh37"
    GRCH38 = "GRCh38"


class VariantType(str, Enum):
    """Types of genetic variants."""
    SNV = "snv"                  # Single nucleotide variant
    INSERTION = "insertion"
    DELETION = "deletion"
    INDEL = "indel"
    DUPLICATION = "duplication"
    FRAMESHIFT = "frameshift"
    MNV = "mnv"                  # Multi-nucleotide variant
    UNKNOWN = "unknown"


class MolecularConsequence(str, Enum):
    """Predicted molecular consequence of variant on transcript/protein."""
    MISSENSE = "missense_variant"
    NONSENSE = "stop_gained"
    FRAMESHIFT = "frameshift_variant"
    SPLICE_DONOR = "splice_donor_variant"
    SPLICE_ACCEPTOR = "splice_acceptor_variant"
    SYNONYMOUS = "synonymous_variant"
    START_LOST = "start_lost"
    STOP_LOST = "stop_lost"
    INFRAME_INSERTION = "inframe_insertion"
    INFRAME_DELETION = "inframe_deletion"
    INTRONIC = "intron_variant"
    UTR_5 = "5_prime_UTR_variant"
    UTR_3 = "3_prime_UTR_variant"
    INTERGENIC = "intergenic_variant"
    UPSTREAM = "upstream_gene_variant"
    DOWNSTREAM = "downstream_gene_variant"
    UNKNOWN = "unknown"


class ClinicalSignificance(str, Enum):
    """ACMG/AMP 5-tier classification system."""
    PATHOGENIC = "Pathogenic"
    LIKELY_PATHOGENIC = "Likely pathogenic"
    UNCERTAIN = "Uncertain significance"
    LIKELY_BENIGN = "Likely benign"
    BENIGN = "Benign"
    CONFLICTING = "Conflicting classifications"
    NOT_PROVIDED = "not provided"


class ACMGCriterion(str, Enum):
    """ACMG/AMP evidence criteria codes."""
    # Very strong pathogenic
    PVS1 = "PVS1"
    # Strong pathogenic
    PS1 = "PS1"; PS2 = "PS2"; PS3 = "PS3"; PS4 = "PS4"
    # Moderate pathogenic
    PM1 = "PM1"; PM2 = "PM2"; PM3 = "PM3"; PM4 = "PM4"; PM5 = "PM5"; PM6 = "PM6"
    # Supporting pathogenic
    PP1 = "PP1"; PP2 = "PP2"; PP3 = "PP3"; PP4 = "PP4"; PP5 = "PP5"
    # Stand-alone benign
    BA1 = "BA1"
    # Strong benign
    BS1 = "BS1"; BS2 = "BS2"; BS3 = "BS3"; BS4 = "BS4"
    # Supporting benign
    BP1 = "BP1"; BP2 = "BP2"; BP3 = "BP3"; BP4 = "BP4"
    BP5 = "BP5"; BP6 = "BP6"; BP7 = "BP7"


@dataclass
class GenomicPosition:
    """A position in the genome."""
    chromosome: str
    position: int
    ref: str
    alt: str
    assembly: Assembly = Assembly.GRCH38

    @property
    def chrom_normalized(self) -> str:
        """Return chromosome without 'chr' prefix."""
        return self.chromosome.replace("chr", "")

    def __str__(self) -> str:
        return f"chr{self.chrom_normalized}:{self.position}:{self.ref}>{self.alt}"


@dataclass
class TranscriptVariant:
    """Variant described at transcript (cDNA) level."""
    transcript_id: str           # e.g., NM_007294.4
    gene_symbol: str             # e.g., BRCA1
    hgvs_c: str                  # e.g., c.5266dupC
    hgvs_p: Optional[str] = None # e.g., p.Gln1756Profs*74
    exon: Optional[int] = None
    consequence: MolecularConsequence = MolecularConsequence.UNKNOWN


@dataclass
class ParsedVariant:
    """Fully parsed and normalized variant with all identifiers."""
    # Primary identifiers (at least one must be present)
    genomic: Optional[GenomicPosition] = None
    transcript: Optional[TranscriptVariant] = None
    rs_id: Optional[str] = None                   # e.g., rs80357906
    clinvar_id: Optional[int] = None              # ClinVar variation ID
    
    # Derived
    gene_symbol: Optional[str] = None
    variant_type: VariantType = VariantType.UNKNOWN
    hgvs_genomic: Optional[str] = None            # g. notation
    
    # Input tracking
    raw_input: str = ""
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Human-readable variant name."""
        if self.transcript and self.transcript.gene_symbol:
            change = self.transcript.hgvs_c or self.transcript.hgvs_p or ""
            return f"{self.transcript.gene_symbol} {change}".strip()
        if self.gene_symbol and self.transcript:
            change = self.transcript.hgvs_c or self.transcript.hgvs_p or ""
            return f"{self.gene_symbol} {change}".strip()
        if self.rs_id:
            return self.rs_id
        if self.genomic:
            return str(self.genomic)
        return self.raw_input

    @property
    def is_valid(self) -> bool:
        """Check if at least one identifier was successfully parsed."""
        return any([self.genomic, self.transcript, self.rs_id, self.clinvar_id])


@dataclass
class PopulationFrequency:
    """Allele frequency in a specific population."""
    population: str          # e.g., "gnomAD_AFR", "gnomAD_NFE"
    frequency: float
    allele_count: int = 0
    allele_number: int = 0
    homozygote_count: int = 0


@dataclass 
class ClinVarRecord:
    """A ClinVar submission record."""
    variation_id: int
    clinical_significance: ClinicalSignificance
    review_status: str                    # e.g., "reviewed by expert panel"
    review_stars: int = 0                 # 0-4
    condition: str = ""
    submitter: str = ""
    last_evaluated: Optional[str] = None
    accession: str = ""                   # SCV accession


@dataclass
class AlphaMissenseScore:
    """AlphaMissense pathogenicity prediction."""
    score: float                          # 0-1, higher = more pathogenic
    classification: str                   # "likely_benign", "ambiguous", "likely_pathogenic"
    transcript_id: str = ""
    protein_change: str = ""


@dataclass
class ProteinDomain:
    """A protein functional domain."""
    name: str
    start: int                            # amino acid position
    end: int
    source: str = ""                      # e.g., "Pfam", "InterPro"
    description: str = ""


@dataclass
class LiteratureEvidence:
    """A PubMed article relevant to the variant."""
    pmid: str
    title: str
    authors: str
    journal: str
    year: int
    abstract: str = ""
    relevance_score: float = 0.0          # From RAG retrieval
    evidence_strength: str = ""           # "strong", "moderate", "limited"


@dataclass
class ACMGEvidence:
    """An applied ACMG/AMP evidence criterion."""
    criterion: ACMGCriterion
    met: bool
    strength: str = "default"             # "default", "strong", "moderate", "supporting"
    reason: str = ""


@dataclass
class VariantReport:
    """Complete variant interpretation report — the final output."""
    # Core variant info
    variant: ParsedVariant
    
    # Database annotations
    clinvar_records: list[ClinVarRecord] = field(default_factory=list)
    population_frequencies: list[PopulationFrequency] = field(default_factory=list)
    alphamissense: Optional[AlphaMissenseScore] = None
    protein_domains: list[ProteinDomain] = field(default_factory=list)
    
    # ML predictions
    pathogenicity_score: Optional[float] = None     # 0-1, our ensemble score
    pathogenicity_label: Optional[str] = None       # "Pathogenic", "VUS", "Benign"
    prediction_confidence: Optional[float] = None
    feature_importances: dict[str, float] = field(default_factory=dict)  # SHAP
    
    # ACMG classification
    acmg_evidence: list[ACMGEvidence] = field(default_factory=list)
    acmg_classification: Optional[ClinicalSignificance] = None
    
    # Literature
    literature: list[LiteratureEvidence] = field(default_factory=list)
    
    # LLM narrative
    narrative_summary: str = ""
    clinical_implications: str = ""
    
    # Metadata
    pipeline_version: str = "0.1.0"
    processing_time_seconds: float = 0.0

    @property
    def max_gnomad_frequency(self) -> Optional[float]:
        """Highest allele frequency across all populations."""
        gnomad = [pf for pf in self.population_frequencies if "gnomAD" in pf.population]
        return max((pf.frequency for pf in gnomad), default=None) if gnomad else None

    @property
    def clinvar_consensus(self) -> Optional[ClinicalSignificance]:
        """Most common ClinVar classification."""
        if not self.clinvar_records:
            return None
        from collections import Counter
        counts = Counter(r.clinical_significance for r in self.clinvar_records)
        return counts.most_common(1)[0][0]
