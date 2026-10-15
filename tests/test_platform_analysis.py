"""Comparison/report-data/legacy-registration checks for the platform CLI."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from multiproposal.platform.execution import RunStatus
from multiproposal.platform.registry import Registry
from tests.test_platform_registry import polar_spec, provider_installed, run_cli, wait_for


@unittest.skipUnless(provider_installed(), "polar-twist provider is not installed")
class AnalysisCommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "runs"
        self.spec_path = Path(self.directory.name) / "spec.json"

    def tearDown(self):
        self.directory.cleanup()

    def submit(self, spec):
        self.spec_path.write_text(spec.to_json())
        result = run_cli("submit", str(self.spec_path), root=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        execution_id = json.loads(result.stdout)["execution_id"]
        registry = Registry(self.root)
        record = wait_for(registry, execution_id, {RunStatus.COMPLETED, RunStatus.FAILED})
        self.assertEqual(record.status, RunStatus.COMPLETED, record.error)
        return execution_id, record

    def test_report_data_returns_costs_and_component_summaries(self):
        execution_id, _ = self.submit(polar_spec(chains=4, draws=30, warmup=10))
        result = run_cli("report-data", execution_id, root=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["execution_id"], execution_id)
        self.assertIn("sampler_0", payload["samplers"])
        variable = payload["samplers"]["sampler_0"]["variables"]["x"]
        self.assertIn("[0]", variable["components"])
        self.assertEqual(variable["components"]["[0]"]["diagnostics"]["status"], "ok")
        self.assertIn("sampler_0", payload["costs"])

    def test_compare_reports_component_deltas(self):
        first, _ = self.submit(polar_spec(chains=3, draws=20, warmup=5))
        second, _ = self.submit(polar_spec(chains=3, draws=20, warmup=5))
        result = run_cli("compare", first, second, root=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["baseline_execution_id"], first)
        delta = payload["comparisons"][second]["sampler_0"]["variables"]["x"]["components"]["[0]"]
        self.assertAlmostEqual(delta["mean_delta"], 0.0, places=12)
        self.assertAlmostEqual(delta["std_delta"], 0.0, places=12)

        invalid = run_cli("compare", first, root=self.root)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("at least two execution IDs", invalid.stderr)

    def test_register_legacy_adds_completed_manifest_to_new_registry(self):
        execution_id, record = self.submit(polar_spec(chains=2, draws=10, warmup=4))
        source = Path(record.run_root)

        with tempfile.TemporaryDirectory() as other:
            legacy_root = Path(other) / "runs"
            legacy_dir = legacy_root / source.relative_to(self.root)
            legacy_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, legacy_dir)
            result = run_cli(
                "register-legacy",
                str(legacy_dir / "manifest.json"),
                root=legacy_root,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["execution_id"], execution_id)
            self.assertEqual(payload["status"], "completed")


if __name__ == "__main__":
    unittest.main()
