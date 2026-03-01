"""Tests for variant parser — covers all supported input formats."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from genomicsgpt.variant_parser.parser import parse_variant, parse_variants_batch
from genomicsgpt.models import (
    Assembly,
    MolecularConsequence,
    VariantType,
)


class TestRsIDParsing:
    """Test rsID format parsing."""

    def test_standard_rsid(self):
        v = parse_variant("rs80357906")
        assert v.is_valid
        assert v.rs_id == "rs80357906"

    def test_rsid_case_insensitive(self):
        v = parse_variant("RS12345")
        assert v.rs_id == "rs12345"

    def test_rsid_large_number(self):
        v = parse_variant("rs1489788442")
        assert v.rs_id == "rs1489788442"


class TestHGVSTranscriptParsing:
    """Test HGVS c. notation parsing."""

    def test_snv(self):
        v = parse_variant("NM_007294.4:c.5123C>A")
        assert v.is_valid
        assert v.transcript is not None
        assert v.transcript.transcript_id == "NM_007294.4"
        assert v.transcript.hgvs_c == "c.5123C>A"
        assert v.variant_type == VariantType.SNV

    def test_duplication(self):
        v = parse_variant("NM_007294.4:c.5266dupC")
        assert v.transcript.hgvs_c == "c.5266dupC"
        assert v.variant_type == VariantType.DUPLICATION

    def test_deletion(self):
        v = parse_variant("NM_000492.3:c.1521_1523delCTT")
        assert v.transcript.hgvs_c == "c.1521_1523delCTT"
        assert v.variant_type == VariantType.DELETION

    def test_insertion(self):
        v = parse_variant("NM_000059.4:c.5946delT")
        assert v.variant_type == VariantType.DELETION

    def test_indel(self):
        v = parse_variant("NM_004333.6:c.1799_1800delinsAA")
        assert v.variant_type == VariantType.INDEL

    def test_without_version(self):
        v = parse_variant("NM_007294:c.5266dupC")
        assert v.transcript.transcript_id == "NM_007294"

    def test_rna_transcript(self):
        v = parse_variant("NR_046018.2:c.100A>G")
        assert v.transcript.transcript_id == "NR_046018.2"


class TestGenomicCoordinateParsing:
    """Test genomic coordinate format parsing."""

    def test_chr_colon_format(self):
        v = parse_variant("chr17:43057051:A>T")
        assert v.is_valid
        assert v.genomic is not None
        assert v.genomic.chromosome == "17"
        assert v.genomic.position == 43057051
        assert v.genomic.ref == "A"
        assert v.genomic.alt == "T"

    def test_no_chr_prefix(self):
        v = parse_variant("17:43057051:A>T")
        assert v.genomic.chromosome == "17"

    def test_dash_format(self):
        v = parse_variant("17-43057051-A-T")
        assert v.genomic is not None
        assert v.genomic.ref == "A"
        assert v.genomic.alt == "T"

    def test_x_chromosome(self):
        v = parse_variant("chrX:153764217:C>T")
        assert v.genomic.chromosome == "X"

    def test_deletion_coordinates(self):
        v = parse_variant("chr7:117559590:ATCT>A")
        assert v.genomic.ref == "ATCT"
        assert v.genomic.alt == "A"
        assert v.variant_type == VariantType.DELETION

    def test_insertion_coordinates(self):
        v = parse_variant("chr17:43057051:A>AGCT")
        assert v.variant_type == VariantType.INSERTION

    def test_snv_type(self):
        v = parse_variant("chr17:43057051:A>T")
        assert v.variant_type == VariantType.SNV


class TestGeneAndMutationParsing:
    """Test gene + mutation shorthand formats."""

    def test_gene_cdna(self):
        v = parse_variant("BRCA1 c.5266dupC")
        assert v.is_valid
        assert v.gene_symbol == "BRCA1"
        assert v.transcript is not None
        assert v.transcript.hgvs_c == "c.5266dupC"
        assert v.transcript.gene_symbol == "BRCA1"

    def test_gene_protein_short(self):
        v = parse_variant("BRAF V600E")
        assert v.gene_symbol == "BRAF"
        assert v.transcript.hgvs_p == "p.Val600Glu"
        assert v.variant_type == VariantType.SNV

    def test_gene_protein_three_letter(self):
        v = parse_variant("TP53 p.Arg175His")
        assert v.gene_symbol == "TP53"
        assert v.transcript.hgvs_p == "p.Arg175His"

    def test_gene_stop_gain(self):
        v = parse_variant("APC R1450*")
        assert v.gene_symbol == "APC"
        assert v.transcript.hgvs_p == "p.Arg1450Ter"

    def test_gene_with_number(self):
        v = parse_variant("BRCA2 c.6174delT")
        assert v.gene_symbol == "BRCA2"

    def test_warns_about_missing_transcript(self):
        v = parse_variant("BRCA1 c.5266dupC")
        assert len(v.parse_warnings) > 0
        assert "Transcript ID" in v.parse_warnings[0]


class TestClinVarIDParsing:
    """Test ClinVar accession parsing."""

    def test_vcv_format(self):
        v = parse_variant("VCV000012345")
        assert v.clinvar_id == 12345

    def test_clinvar_prefix(self):
        v = parse_variant("ClinVar:12345")
        assert v.clinvar_id == 12345

    def test_clinvar_with_space(self):
        v = parse_variant("clinvar 54321")
        assert v.clinvar_id == 54321


class TestConsequenceInference:
    """Test molecular consequence inference."""

    def test_frameshift_from_dup(self):
        v = parse_variant("NM_007294.4:c.5266dupC")
        assert v.transcript.consequence == MolecularConsequence.FRAMESHIFT

    def test_missense_snv(self):
        v = parse_variant("NM_004333.6:c.1799T>A")
        assert v.transcript.consequence == MolecularConsequence.MISSENSE

    def test_splice_donor(self):
        v = parse_variant("NM_000059.4:c.7007+1G>A")
        assert v.transcript.consequence == MolecularConsequence.SPLICE_DONOR

    def test_splice_acceptor(self):
        v = parse_variant("NM_000059.4:c.7008-2A>G")
        assert v.transcript.consequence == MolecularConsequence.SPLICE_ACCEPTOR


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_input(self):
        v = parse_variant("")
        assert not v.is_valid
        assert len(v.parse_warnings) > 0

    def test_whitespace_only(self):
        v = parse_variant("   ")
        assert not v.is_valid

    def test_garbage_input(self):
        v = parse_variant("hello world this is not a variant")
        assert not v.is_valid
        assert "Could not parse" in v.parse_warnings[0]

    def test_display_name_rsid(self):
        v = parse_variant("rs80357906")
        assert v.display_name == "rs80357906"

    def test_display_name_gene(self):
        v = parse_variant("BRCA1 c.5266dupC")
        assert "BRCA1" in v.display_name
        assert "c.5266dupC" in v.display_name

    def test_display_name_coordinates(self):
        v = parse_variant("chr17:43057051:A>T")
        assert "17" in v.display_name

    def test_raw_input_preserved(self):
        v = parse_variant("rs80357906")
        assert v.raw_input == "rs80357906"


class TestBatchParsing:
    """Test batch variant parsing."""

    def test_mixed_formats(self):
        inputs = [
            "rs80357906",
            "BRCA1 c.5266dupC",
            "chr17:43057051:A>T",
            "BRAF V600E",
        ]
        results = parse_variants_batch(inputs)
        assert len(results) == 4
        assert all(v.is_valid for v in results)
        assert results[0].rs_id is not None
        assert results[1].gene_symbol == "BRCA1"
        assert results[2].genomic is not None
        assert results[3].transcript.hgvs_p == "p.Val600Glu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
