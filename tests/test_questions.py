import unittest

from financial_analyst.questions import UnsupportedQuestion, plan_question


class QuestionTests(unittest.TestCase):
    def plan(self, question):
        return plan_question(question, "Example Pharma", "1QFY27")

    def test_supported_complete_question(self):
        plan = self.plan("By how much did Example Pharma's 1QFY27 revenue exceed the broker's estimate, and what was the YoY growth?")
        self.assertEqual(plan.operation, "compare")
        self.assertTrue(plan.include_yoy)

    def test_growth_and_beat_causality_have_different_routes(self):
        self.assertEqual(self.plan("Why did Example Pharma's 1QFY27 revenue grow?").operation, "growth")
        self.assertEqual(self.plan("Why did Example Pharma's 1QFY27 revenue beat the estimate?").operation, "beat_attribution")

    def test_mixed_or_unsupported_requests_are_not_partially_answered(self):
        questions = [
            "Compare Example Pharma and Other Pharma 1QFY27 revenue",
            "What was Example Pharma 1QFY27 revenue and EBITDA?",
            "By how much did Example Pharma 1QFY27 revenue beat the estimate and what was PAT?",
            "Did Example Pharma 1QFY27 revenue not beat the estimate?",
            "What was Example Pharma 2QFY27 YoY growth?",
            "When does Example Pharma FY27 start and end?",
            "Did Example Pharma 1QFY27 standalone revenue beat the estimate?",
        ]
        for question in questions:
            with self.subTest(question=question), self.assertRaises(UnsupportedQuestion):
                self.plan(question)


if __name__ == "__main__":
    unittest.main()
