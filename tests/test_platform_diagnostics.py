"""Diagnostics contract tests for ArviZ-backed platform summaries."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multiproposal.platform.diagnostics import summarize_chain_array
from multiproposal.platform.registry import Registry
from tests.test_platform_registry import polar_spec, provider_installed, run_cli, wait_for
from multiproposal.platform.execution import RunStatus


class ArrayDiagnosticsTests(unittest.TestCase):
    def test_unavailable_reasons_are_explicit(self):
        one_chain = summarize_chain_array(np.array([[1.0, 2.0, 3.0, 4.0]]))
        self.assertEqual(one_chain["status"], "unavailable")
        self.assertTrue(any("one-chain" in warning for warning in one_chain["warnings"]))

        short = summarize_chain_array(np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]))
        self.assertEqual(short["status"], "unavailable")
        self.assertTrue(any("too-short" in warning for warning in short["warnings"]))

        constant = summarize_chain_array(np.ones((2, 5)))
        self.assertEqual(constant["status"], "unavailable")
        self.assertTrue(any("constant" in warning for warning in constant["warnings"]))

        nonfinite = summarize_chain_array(np.array([[1.0, 2.0, np.nan, 4.0], [1.2, 2.3, 3.4, 4.5]]))
        self.assertEqual(nonfinite["status"], "unavailable")
        self.assertTrue(any("nonfinite" in warning for warning in nonfinite["warnings"]))

    def test_ok_case_returns_metrics(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(4, 30))
        summary = summarize_chain_array(values)
        self.assertEqual(summary["status"], "ok")
        for key in ("rhat_rank", "rhat_folded", "ess_bulk", "ess_tail", "mcse_mean"):
            self.assertIsInstance(summary["metrics"][key], float)


@unittest.skipUnless(provider_installed(), "polar-twist provider is not installed")
class DiagnoseCommandTests(unittest.TestCase):
    def test_diagnose_reads_completed_worker_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            spec_path = Path(directory) / "spec.json"
            spec_path.write_text(polar_spec(chains=4, draws=30, warmup=10).to_json())

            submitted = run_cli("submit", str(spec_path), root=root)
            self.assertEqual(submitted.returncode, 0, submitted.stderr)
            execution_id = json.loads(submitted.stdout)["execution_id"]

            registry = Registry(root)
            record = wait_for(registry, execution_id, {RunStatus.COMPLETED, RunStatus.FAILED})
            self.assertEqual(record.status, RunStatus.COMPLETED, record.error)

            result = run_cli("diagnose", execution_id, root=root)
            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(result.stdout)
            variable = payload["samplers"]["sampler_0"]["variables"]["x"]
            component = variable["components"]["[0]"]
            self.assertEqual(component["status"], "ok")
            self.assertGreater(component["metrics"]["ess_bulk"], 0.0)


if __name__ == "__main__":
    unittest.main()
