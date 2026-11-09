import copy
import json
import tempfile
import unittest
from pathlib import Path

from deepshield.evaluation import load_transfer_protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "configs" / "transfer_protocol.2026-11-09.v1.json"


class TransferProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def _load(self, value):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            return load_transfer_protocol(path)

    def test_reference_protocol_is_blocked_and_counted(self):
        plan = load_transfer_protocol(PROTOCOL_PATH).plan()
        self.assertFalse(plan.execution_ready)
        self.assertEqual(plan.evidence_state, "unrun")
        self.assertEqual(plan.total_editor_calls, 234)
        self.assertEqual(plan.total_protection_jobs, 36)
        self.assertEqual(plan.first_gate_gpu_hours, 1.267)
        self.assertIn("rights_cleared_assets_missing", plan.blockers)
        self.assertIn("compute_not_approved", plan.blockers)

    def test_cost_arithmetic_requires_explicit_price(self):
        protocol = load_transfer_protocol(PROTOCOL_PATH)
        without_price = protocol.plan()
        with_price = protocol.plan(2.0)
        self.assertIsNone(without_price.estimated_gpu_cost_usd)
        self.assertEqual(with_price.estimated_gpu_cost_usd, 33.8)
        self.assertEqual(with_price.first_gate_cost_usd, 2.53)

    def test_rejects_non_unrun_preparation_evidence(self):
        value = copy.deepcopy(self.reference)
        value["evidence_state"] = "measured"
        with self.assertRaisesRegex(ValueError, "evidence_state='unrun'"):
            self._load(value)

    def test_rejects_unapproved_asset(self):
        value = copy.deepcopy(self.reference)
        value["asset_policy"]["assets"] = [{
            "id": "image-1",
            "split": "development",
            "authorized": False,
            "source_type": "synthetic",
            "provenance_url": "https://example.invalid/source",
            "license_or_permission": "unknown",
            "sha256": "0" * 64,
        }]
        with self.assertRaisesRegex(ValueError, "explicitly authorized"):
            self._load(value)

    def test_rejects_editor_leakage_across_splits(self):
        value = copy.deepcopy(self.reference)
        value["stages"][1]["editor_ids"] = ["flux2-klein-held-out"]
        with self.assertRaisesRegex(ValueError, "does not match its split"):
            self._load(value)

    def test_rejects_unknown_preprocessing(self):
        value = copy.deepcopy(self.reference)
        value["stages"][0]["preprocessing_ids"] = ["unregistered"]
        with self.assertRaisesRegex(ValueError, "unknown preprocessing"):
            self._load(value)

    def test_unrun_cells_cannot_be_encoded_as_zero(self):
        value = copy.deepcopy(self.reference)
        value["outcomes"]["initial_cell_state"] = 0
        with self.assertRaisesRegex(ValueError, "initial_cell_state must be 'unrun'"):
            self._load(value)

    def test_preparation_cannot_self_approve_compute(self):
        value = copy.deepcopy(self.reference)
        value["compute"]["first_gate_proposal"]["approval_status"] = "approved"
        with self.assertRaisesRegex(ValueError, "cannot approve compute"):
            self._load(value)


if __name__ == "__main__":
    unittest.main()
