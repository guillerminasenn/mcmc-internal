"""Diagnostics contract tests for ArviZ-backed platform summaries."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from multiproposal.platform.diagnostics import summarize_chain_array, summarize_diagnostic_evolution
from multiproposal.platform.registry import Registry
from tests.test_platform_registry import polar_spec, provider_installed, run_cli, wait_for
from multiproposal.platform.execution import RunStatus


class ArrayDiagnosticsTests(unittest.TestCase):
    def test_evolution_preserves_chain_axes_and_bounds_default_points(self):
        values = np.random.default_rng(4).normal(size=(3, 100))
        with patch("multiproposal.platform.diagnostics.summarize_chain_array",
                   wraps=summarize_chain_array) as summarize:
            evolution = summarize_diagnostic_evolution(values)
        prefixes = [point["draws"] for point in evolution["points"]]
        self.assertEqual((prefixes[0], prefixes[-1]), (1, 100))
        self.assertLessEqual(len(prefixes), 20)
        self.assertEqual([call.args[0].shape for call in summarize.call_args_list],
                         [(3, prefix) for prefix in prefixes])
        self.assertEqual(evolution["chain_count"], 3)
        self.assertEqual(evolution["retained_draws"], 100)

    def test_evolution_distinguishes_iid_and_autocorrelated_fixtures(self):
        rng = np.random.default_rng(42)
        iid = rng.normal(size=(4, 400))
        noise = rng.normal(size=(4, 400))
        autocorrelated = np.empty_like(noise)
        autocorrelated[:, 0] = noise[:, 0]
        for draw in range(1, noise.shape[1]):
            autocorrelated[:, draw] = 0.95 * autocorrelated[:, draw - 1] + noise[:, draw]
        prefixes = (3, 20, 100, 400)
        iid_points = summarize_diagnostic_evolution(iid, prefixes=prefixes)["points"]
        autocorrelated_points = summarize_diagnostic_evolution(
            autocorrelated, prefixes=prefixes,
        )["points"]
        self.assertEqual(iid_points[0]["availability"]["ess_bulk"], "too-short")
        self.assertIsNone(iid_points[0]["metrics"]["mcse_mean"])
        self.assertGreater(iid_points[-1]["metrics"]["ess_bulk"],
                           10 * autocorrelated_points[-1]["metrics"]["ess_bulk"])
        self.assertLess(iid_points[-1]["metrics"]["mcse_mean"],
                        autocorrelated_points[-1]["metrics"]["mcse_mean"] / 10)

    def test_evolution_rejects_invalid_shapes_prefixes_and_bounds(self):
        values = np.ones((2, 8))
        for invalid in ((), (0,), (1, 9), (4, 4), (5, 3), (True,)):
            with self.subTest(prefixes=invalid), self.assertRaises(ValueError):
                summarize_diagnostic_evolution(values, prefixes=invalid)
        for invalid in (0, True, 1.5):
            with self.subTest(max_points=invalid), self.assertRaises(ValueError):
                summarize_diagnostic_evolution(values, max_points=invalid)
        with self.assertRaises(ValueError):
            summarize_diagnostic_evolution(np.ones((2, 0)))

    def test_one_chain_retains_supported_ess_and_mcse(self):
        values = np.random.default_rng(4).normal(size=(1, 30))
        summary = summarize_chain_array(values)
        for name in ("rhat_rank", "rhat_folded"):
            self.assertIsNone(summary["metrics"][name])
            self.assertEqual(summary["availability"][name], "one-chain")
        for name in ("ess_bulk", "ess_tail", "mcse_mean"):
            self.assertTrue(np.isfinite(summary["metrics"][name]))
            self.assertIsNone(summary["availability"][name])

    def test_empty_short_nonfinite_and_within_chain_constants(self):
        for values, reason in ((np.empty((2, 0)), "too-short"), (np.ones((0, 5)), "too-short"),
                               (np.arange(6).reshape(2, 3), "too-short"),
                               (np.repeat([[1.], [2.]], 8, axis=1), "constant"),
                               (np.full((2, 8), np.inf), "nonfinite")):
            with self.subTest(reason=reason):
                summary = summarize_chain_array(values)
                self.assertTrue(all(value is None for value in summary["metrics"].values()))
                self.assertEqual(set(summary["availability"].values()), {reason})
                json.dumps(summary, allow_nan=False)

    def test_metric_nonfinite_result_does_not_hide_other_metrics(self):
        values = np.random.default_rng(4).normal(size=(2, 30))
        with patch("arviz.rhat", return_value=np.nan):
            summary = summarize_chain_array(values)
        self.assertEqual(summary["availability"]["rhat_rank"], "nonfinite-result")
        self.assertIsNone(summary["metrics"]["rhat_rank"])
        self.assertIsInstance(summary["metrics"]["ess_bulk"], float)
        json.dumps(summary, allow_nan=False)

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
