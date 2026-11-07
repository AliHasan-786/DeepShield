import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from deepshield.evaluation import (
    EvaluationResult,
    build_matrix_markdown,
    load_manifest,
)
from deepshield.evaluation.preprocessing import jpeg_reencode
from deepshield.evaluation.metrics import MetricScores
from deepshield.evaluation.runner import run_evaluation


class FakeAdapter:
    def __init__(self):
        self.calls = []
        self.closed = 0

    def run(self, spec, image, mask, prompt, seed):
        self.calls.append((spec.id, seed, prompt, image.size, mask.size))
        return Image.new("RGB", image.size, (seed, 10, 20))

    def close(self):
        self.closed += 1


class FakeMetrics:
    def score(self, identity_reference, original, protected, attacked, prompt):
        return MetricScores(0.8, 0.4, 0.1)


class EvaluationSchemaTests(unittest.TestCase):
    def _manifest(self):
        return {
            "schema_version": 1,
            "protocol_status": "draft",
            "seeds": [1, 2],
            "thresholds": {
                "identity_similarity_min": 0.6,
                "clip_edit_fidelity_min": 0.2,
                "source": "test fixture",
            },
            "samples": [{
                "id": "sample",
                "original_path": "original.png",
                "mask_path": "mask.png",
                "prompt": "make the jacket blue",
                "identity_reference_path": "original.png",
                "variants": [{
                    "id": "e20",
                    "path": "protected.png",
                    "method": "DeepShield",
                    "epsilon_255": 20,
                }],
            }],
            "pipelines": [{
                "id": "sd15",
                "architecture": "SD1.5",
                "task": "inpainting",
                "model_id": "example/model",
                "jpeg_quality": 85,
                "steps": 20,
                "guidance_scale": 7.5,
                "width": 512,
                "height": 512,
                "strength": 0.75,
            }],
        }

    def test_loads_and_counts_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            manifest = load_manifest(path)
        self.assertEqual(manifest.planned_run_count, 2)

    def test_rejects_invalid_jpeg_quality(self):
        value = self._manifest()
        value["pipelines"][0]["jpeg_quality"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "jpeg_quality"):
                load_manifest(path)

    def test_jpeg_reencode_is_deterministic_and_sized(self):
        image = Image.new("RGBA", (23, 17), (20, 40, 60, 120))
        first = jpeg_reencode(image, 85, (16, 16))
        second = jpeg_reencode(image, 85, (16, 16))
        self.assertEqual(first.size, (16, 16))
        self.assertEqual(first.mode, "RGB")
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_matrix_keeps_failed_protection_visible(self):
        base = dict(
            sample_id="sample",
            variant_id="e20",
            method="DeepShield",
            epsilon_255=20,
            pipeline_id="sd15",
            architecture="SD1.5",
            jpeg_quality=85,
            identity_similarity=0.8,
            clip_edit_fidelity=0.4,
            protection_lpips=0.1,
            original_output_path="original.png",
            protected_output_path="protected.png",
        )
        results = [
            EvaluationResult(seed=1, attack_succeeded=True, **base),
            EvaluationResult(seed=2, attack_succeeded=True, **base),
        ]
        matrix = build_matrix_markdown(results)
        self.assertIn("100% (2 runs)", matrix)
        self.assertIn("protection failed", matrix)

    def test_empty_matrix_does_not_imply_results(self):
        self.assertIn("not evidence", build_matrix_markdown([]))

    def test_runner_pairs_inputs_and_persists_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, color in {
                "original.png": (10, 20, 30),
                "protected.png": (11, 21, 31),
                "mask.png": (255, 255, 255),
            }.items():
                Image.new("RGB", (20, 20), color).save(root / name)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(self._manifest()), encoding="utf-8")
            adapter = FakeAdapter()
            output_dir = root / "outputs"
            results = run_evaluation(
                load_manifest(manifest_path),
                root,
                output_dir,
                adapter,
                FakeMetrics(),
            )
            lines = (output_dir / "results.jsonl").read_text().splitlines()

        self.assertEqual(len(results), 2)
        self.assertEqual(len(lines), 2)
        self.assertEqual([call[1] for call in adapter.calls], [1, 1, 2, 2])
        self.assertEqual(adapter.closed, 1)
        self.assertTrue(all(result.attack_succeeded for result in results))


if __name__ == "__main__":
    unittest.main()
