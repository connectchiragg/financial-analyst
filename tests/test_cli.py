from decimal import Decimal, Inexact, localcontext
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from financial_analyst.cli import _render_text
from support import create_fixture


PROJECT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "financial_analyst", *args],
                              cwd=PROJECT, capture_output=True, text=True, check=False)

    def command(self, *args, output="json"):
        return self.run_cli("--mode", "fixture", "--fixture", str(self.fixture),
                            "--format", output, *args)

    def test_combined_json_preserves_originals_citations_and_modes(self):
        run = self.command("combined", "--company", "Example Pharma", "--period", "1QFY27", "--yoy")
        self.assertEqual(run.returncode, 0, run.stderr)
        answer = json.loads(run.stdout)
        self.assertEqual(answer["status"], "answered")
        comparison, yoy = answer["claims"][:2]
        self.assertEqual(Decimal(comparison["values"]["delta"]), Decimal("20"))
        self.assertEqual(Decimal(comparison["values"]["beat_percent"]), Decimal("20"))
        self.assertEqual(Decimal(yoy["values"]["yoy_percent"]), Decimal("50"))
        self.assertEqual(answer["comparison"]["actual"]["source_header_raw"], "FY27E / 1Q")
        citation_refs = {citation["ref"] for citation in answer["citations"]}
        for claim in answer["claims"]:
            self.assertTrue(set(claim["evidence_refs"]) <= citation_refs)
        self.assertTrue({"status", "growth"} <= set(comparison["evidence_refs"]))
        self.assertEqual(answer["execution"]["retrieval"], "curated_fixture")
        self.assertTrue(answer["execution"]["no_llm"])
        self.assertTrue(answer["execution"]["no_database"])

    def test_text_labels_source_context_and_requested_retrieval_truthfully(self):
        run = self.command("compare", "--company", "Example Pharma", "--period", "1QFY27", output="text")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("retrieval: not requested", run.stdout)
        self.assertIn("FY27E / 1Q", run.stdout)
        self.assertIn("consolidated", run.stdout)
        self.assertIn("INRm", run.stdout)
        self.assertIn("synthetic-report.pdf, pp.", run.stdout)
        self.assertIn("Y/E March", run.stdout)
        self.assertIn("Y/E December", run.stdout)
        self.assertIn("Calendar dates remain unresolved", run.stdout)

    def test_attribution_and_unknown_questions_refuse_without_claims(self):
        attribution = self.command("beat-attribution", "--company", "Example Pharma", "--period", "1QFY27")
        self.assertEqual(attribution.returncode, 1, attribution.stderr)
        answer = json.loads(attribution.stdout)
        self.assertEqual(answer["status"], "refused")
        self.assertEqual(answer["claims"], [])
        self.assertIn("allocation", answer["reason"])
        unknown = self.command("ask", "--company", "Example Pharma", "--period", "1QFY27",
                               "What was Example Pharma revenue and PAT?")
        self.assertEqual(unknown.returncode, 1)
        self.assertEqual(json.loads(unknown.stdout)["claims"], [])

    def test_question_route_and_unavailable_capabilities_do_not_fallback(self):
        run = self.command("ask", "--company", "Example Pharma", "--period", "1QFY27",
                           "By how much did Example Pharma's 1QFY27 revenue exceed the broker estimate?")
        self.assertEqual(run.returncode, 0, run.stderr)
        for flags in [("--mode", "live"), ("--mode", "fixture", "--retrieval", "semantic"),
                      ("--mode", "fixture", "--retrieval", "keyword")]:
            unavailable = self.run_cli(*flags, "compare", "--company", "Example Pharma", "--period", "1QFY27")
            self.assertEqual(unavailable.returncode, 2)
            self.assertIn("no fallback", unavailable.stderr)
            self.assertEqual(unavailable.stdout, "")

    def test_explicit_fixture_and_supported_company_period_are_required(self):
        missing = self.run_cli("--mode", "fixture", "compare", "--company", "Example Pharma", "--period", "1QFY27")
        self.assertEqual(missing.returncode, 2)
        for company, period in [("Other Pharma", "1QFY27"), ("Example Pharma", "2QFY27")]:
            run = self.command("compare", "--company", company, "--period", period)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout)["claims"], [])

    def test_exact_miss_display_is_independent_of_decimal_context(self):
        amount = Decimal("123456789012345678901234567890123456789012345")
        citation = SimpleNamespace(ref="cell", document_name="synthetic-report.pdf", page=2, link=None)
        observation = SimpleNamespace(scope="consolidated", source_metric_raw="Net Sales", source_unit_raw="INRm",
                                      year_end_raw="Y/E March", source_header_raw="FY27 / 1Q", kind="reported_actual")
        claim = SimpleNamespace(kind="comparison", evidence_refs=("cell",), values={
            "company": "Example Pharma", "period": "1QFY27", "currency": "INR", "actual": Decimal("0"),
            "estimate": amount, "delta": amount.copy_negate(), "beat_percent": Decimal("-100")})
        answer = SimpleNamespace(status="answered", claims=(claim,), citations=(citation,), source_conflicts=(),
                                 comparison=SimpleNamespace(actual=observation, broker_estimate=observation),
                                 execution={"retrieval": "not_requested"})
        with localcontext() as context:
            context.prec = 6
            context.traps[Inexact] = True
            text = _render_text(answer)
        self.assertIn(f"missed by {format(amount, ',f')} INR million", text)


if __name__ == "__main__":
    unittest.main()
