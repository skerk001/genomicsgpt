"""Tests for the LLM narrative engine."""

import pytest
from unittest.mock import MagicMock, patch

from genomicsgpt.models import (
    ParsedVariant, TranscriptVariant, GenomicPosition,
    ClinVarRecord, PopulationFrequency, VariantReport,
    Assembly, VariantType, MolecularConsequence, ClinicalSignificance,
)
from genomicsgpt.llm_engine.report_generator import (
    ClinicalNarrative,
    ReportGenerator,
    assemble_evidence,
    build_prompt,
    _interpret_frequency,
)


# ── Fixtures ──


@pytest.fixture
def brca1_variant():
    return ParsedVariant(
        genomic=GenomicPosition(chromosome="17", position=43057051, ref="C", alt="CC"),
        transcript=TranscriptVariant(
            transcript_id="NM_007294.4", gene_symbol="BRCA1",
            hgvs_c="c.5266dupC", hgvs_p="p.Gln1756ProfsTer74",
            consequence=MolecularConsequence.FRAMESHIFT,
        ),
        rs_id="rs80357906", gene_symbol="BRCA1",
        variant_type=VariantType.INSERTION, raw_input="BRCA1 c.5266dupC",
    )


@pytest.fixture
def full_report(brca1_variant):
    return VariantReport(
        variant=brca1_variant,
        clinvar_records=[
            ClinVarRecord(
                variation_id=17661,
                clinical_significance=ClinicalSignificance.PATHOGENIC,
                review_status="reviewed by expert panel", review_stars=3,
                condition="Hereditary breast and ovarian cancer syndrome",
                submitter="ENIGMA",
            ),
            ClinVarRecord(
                variation_id=17661,
                clinical_significance=ClinicalSignificance.PATHOGENIC,
                review_status="criteria provided, single submitter", review_stars=1,
                condition="Breast cancer", submitter="GeneDx",
            ),
        ],
        population_frequencies=[
            PopulationFrequency(population="gnomAD_NFE", frequency=0.0000041, allele_count=5, allele_number=121860),
            PopulationFrequency(population="gnomAD_AFR", frequency=0.0, allele_count=0, allele_number=24042),
        ],
        pathogenicity_score=0.9997,
        pathogenicity_label="Pathogenic",
        prediction_confidence=0.998,
        feature_importances={"is_lof": 2.34, "cons_frameshift": 1.89, "gene_path_ratio": 1.52},
    )


@pytest.fixture
def minimal_report():
    return VariantReport(
        variant=ParsedVariant(rs_id="rs12345", raw_input="rs12345"),
    )


# ── Evidence Assembly Tests ──


class TestAssembleEvidence:
    def test_full_evidence(self, full_report):
        ev = assemble_evidence(full_report)
        assert ev["variant"]["display_name"] == "BRCA1 c.5266dupC"
        assert ev["variant"]["gene"] == "BRCA1"
        assert ev["variant"]["rs_id"] == "rs80357906"
        assert ev["clinvar"]["num_records"] == 2
        assert ev["clinvar"]["consensus"] == "Pathogenic"
        assert ev["ml_prediction"]["score"] == 0.9997
        assert len(ev["population"]["frequencies"]) == 2

    def test_minimal_evidence(self, minimal_report):
        ev = assemble_evidence(minimal_report)
        assert ev["variant"]["rs_id"] == "rs12345"
        assert ev["clinvar"] == {}
        assert ev["ml_prediction"] == {}

    def test_ml_features_sorted(self, full_report):
        ev = assemble_evidence(full_report)
        features = list(ev["ml_prediction"]["top_features"].keys())
        assert features[0] == "is_lof"  # Highest abs SHAP value

    def test_clinvar_limited_to_10(self, full_report):
        # Add 15 records
        full_report.clinvar_records *= 8  # Now 16 records
        ev = assemble_evidence(full_report)
        assert len(ev["clinvar"]["records"]) == 10


# ── Frequency Interpretation Tests ──


class TestInterpretFrequency:
    def test_none(self):
        assert "Not observed" in _interpret_frequency(None)

    def test_zero(self):
        assert "Absent" in _interpret_frequency(0)

    def test_extremely_rare(self):
        result = _interpret_frequency(0.000004)
        assert "Extremely rare" in result

    def test_very_rare(self):
        result = _interpret_frequency(0.00005)
        assert "Very rare" in result

    def test_common(self):
        result = _interpret_frequency(0.15)
        assert "Common" in result
        assert "BA1" in result


# ── Prompt Construction Tests ──


class TestBuildPrompt:
    def test_contains_variant_identity(self, full_report):
        ev = assemble_evidence(full_report)
        prompt = build_prompt(ev)
        assert "BRCA1" in prompt
        assert "c.5266dupC" in prompt
        assert "rs80357906" in prompt

    def test_contains_clinvar(self, full_report):
        ev = assemble_evidence(full_report)
        prompt = build_prompt(ev)
        assert "ClinVar Evidence" in prompt
        assert "Pathogenic" in prompt
        assert "expert panel" in prompt

    def test_contains_ml_prediction(self, full_report):
        ev = assemble_evidence(full_report)
        prompt = build_prompt(ev)
        assert "ML Prediction" in prompt
        assert "0.9997" in prompt
        assert "is_lof" in prompt

    def test_contains_population(self, full_report):
        ev = assemble_evidence(full_report)
        prompt = build_prompt(ev)
        assert "Population Frequencies" in prompt
        assert "gnomAD_NFE" in prompt

    def test_minimal_prompt(self, minimal_report):
        ev = assemble_evidence(minimal_report)
        prompt = build_prompt(ev)
        assert "Variant Identity" in prompt
        assert "ClinVar Evidence" not in prompt  # No ClinVar data


# ── Section Parsing Tests ──


class TestParseSections:
    def test_parses_all_sections(self):
        mock_response = """## Variant Summary
This is a frameshift variant in BRCA1.

## Classification
Pathogenic with high confidence.

## Evidence Summary
Multiple ClinVar submissions support pathogenicity.

## Molecular Mechanism
Causes premature stop codon.

## Population Data
Extremely rare in gnomAD.

## Clinical Implications
High-risk for HBOC syndrome.

## ACMG Criteria
PVS1, PM2, PP5 are met.

## Limitations
No functional studies cited."""

        gen = MagicMock(spec=ReportGenerator)
        gen._parse_sections = ReportGenerator._parse_sections.__get__(gen)

        narrative = gen._parse_sections(mock_response)
        assert "frameshift" in narrative.variant_summary
        assert "Pathogenic" in narrative.classification
        assert "ClinVar" in narrative.evidence_summary
        assert "premature stop" in narrative.molecular_mechanism
        assert "gnomAD" in narrative.population_data
        assert "HBOC" in narrative.clinical_implications
        assert "PVS1" in narrative.acmg_criteria
        assert "functional" in narrative.limitations


# ── ClinicalNarrative Tests ──


class TestClinicalNarrative:
    def test_defaults(self):
        n = ClinicalNarrative()
        assert n.variant_summary == ""
        assert n.tokens_used == 0
        assert n.generation_time_seconds == 0.0

    def test_full_report_set(self):
        n = ClinicalNarrative(
            full_report="Test report",
            model_used="claude-sonnet-4-20250514",
            tokens_used=500,
        )
        assert n.full_report == "Test report"
        assert n.tokens_used == 500


# ── ReportGenerator Tests ──


class TestReportGenerator:
    def test_requires_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                ReportGenerator(api_key=None)

    @patch("genomicsgpt.llm_engine.report_generator.ReportGenerator.__init__", return_value=None)
    def test_generate_calls_api(self, mock_init, full_report):
        gen = ReportGenerator.__new__(ReportGenerator)
        gen.model = "claude-sonnet-4-20250514"
        gen.max_tokens = 2000

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="## Variant Summary\nTest")]
        mock_response.usage = MagicMock(input_tokens=100, output_tokens=200)

        gen.client = MagicMock()
        gen.client.messages.create.return_value = mock_response

        narrative = gen.generate(full_report)

        gen.client.messages.create.assert_called_once()
        call_kwargs = gen.client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert "BRCA1" in call_kwargs["messages"][0]["content"]
        assert narrative.tokens_used == 300
