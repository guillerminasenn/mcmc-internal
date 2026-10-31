"""Comparison/report-data/legacy-registration checks for the platform CLI."""

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from multiproposal.platform.analysis import (_compare_reports, bundle_analysis_provenance,
    bundle_report_data, select_chain_array, summarize_efficiency, summarize_sampler_costs,
    summarize_selection)
from multiproposal.platform.diagnostics import summarize_msjd
from multiproposal.platform.execution import RunStatus
from multiproposal.platform.registry import Registry
from tests.test_platform_registry import polar_spec, provider_installed, run_cli, wait_for
from multiproposal.platform.specs import content_hash


class DrawSelectionTests(unittest.TestCase):
    def setUp(self):
        states = np.array([[0., 0.], [0., 0.], [1., 2.], [1., 2.], [3., 2.]])
        self.values = np.stack([states, states + 100.])
        self.values.setflags(write=False)

    def test_exact_cutoffs_and_immutable_source(self):
        before = self.values.tobytes()
        for cutoff in (0, 2, 3, 4):
            with self.subTest(cutoff=cutoff):
                selection = select_chain_array(self.values, cutoff=cutoff, warmup_draws=2, warmup_included=True)
                np.testing.assert_array_equal(selection.values, self.values[:, cutoff:])
                self.assertFalse(selection.values.flags.writeable)
                self.assertEqual(selection.metadata()["retained_samples"], 2 * (5 - cutoff))
                self.assertEqual(selection.warmup_draws, 2)
        self.assertEqual(before, self.values.tobytes())

    def test_defaults_and_original_chain_ids(self):
        np.testing.assert_array_equal(
            select_chain_array(self.values, warmup_draws=2, warmup_included=True).values,
            self.values[:, 2:])
        np.testing.assert_array_equal(
            select_chain_array(self.values, warmup_draws=20).values, self.values)
        selection = select_chain_array(self.values, cutoff=2, chains=(1, 0))
        self.assertEqual(selection.chain_ids, (1, 0))
        np.testing.assert_array_equal(selection.values, self.values[[1, 0], 2:])

    def test_invalid_cutoffs_and_chains(self):
        for cutoff in (-1, 5, True, False, 2., "2", np.bool_(True)):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                select_chain_array(self.values, cutoff=cutoff)
        for chains in ((), (0, 0), (True,), (-1,), (2,), (1.,)):
            with self.subTest(chains=chains), self.assertRaises(ValueError):
                select_chain_array(self.values, chains=chains)

    def test_msjd_includes_repeats_without_chain_boundary_jumps(self):
        for cutoff, expected in ((0, 2.25), (2, 2.), (3, 4.)):
            selection = select_chain_array(self.values, cutoff=cutoff)
            self.assertEqual(summarize_msjd(selection.values), {"value": expected, "reason": None})
        self.assertEqual(summarize_msjd(self.values[:, 4:])["reason"], "too-short")
        self.assertEqual(summarize_msjd(np.ones((2, 5, 2)))["value"], 0.)
        self.assertEqual(summarize_msjd(np.full((2, 5), np.nan))["reason"], "nonfinite")

    def test_shared_summary_uses_only_retained_values_and_scalar_or_none(self):
        for cutoff in (0, 2, 3, 4):
            selection = select_chain_array(self.values, cutoff=cutoff)
            summary = summarize_selection(selection)
            self.assertEqual(summary["components"]["[0]"]["evolution"]["retained_draws"], 5 - cutoff)
            self.assertEqual(summary["diagnostic_summary"]["ess_bulk"]["available"] +
                             summary["diagnostic_summary"]["ess_bulk"]["unavailable"], 2)
            for index, component in enumerate(summary["components"].values()):
                self.assertEqual(component["mean"], float(np.mean(self.values[:, cutoff:, index])))
                self.assertEqual(component["std"], float(np.std(self.values[:, cutoff:, index])))
                self.assertEqual(component["msjd"], summarize_msjd(self.values[:, cutoff:, index])["value"])
            json.dumps(summary, allow_nan=False)
        summary = summarize_selection(select_chain_array(np.full((2, 5), np.inf)))
        self.assertIsNone(summary["components"]["scalar"]["mean"])
        self.assertIsNone(summary["components"]["scalar"]["std"])
        json.dumps(summary, allow_nan=False)

    def test_recorded_cost_scope_and_efficiency_are_explicit(self):
        raw = [
            {"wall_seconds": 2.0, "log_likelihood_evaluations": 20, "acceptance_rate": 0.25},
            {"wall_seconds": 3.0, "log_likelihood_evaluations": 30, "acceptance_rate": 0.5},
        ]
        costs = summarize_sampler_costs(
            raw, (1, 0),
            {"started_at": "2026-10-31T12:00:00+00:00", "finished_at": "2026-10-31T12:00:07+00:00"},
        )
        self.assertTrue(costs["includes_warmup"])
        self.assertEqual(costs["chain_ids"], [1, 0])
        self.assertEqual(costs["totals"], {
            "sampler_wall_seconds": 5.0,
            "log_likelihood_evaluations": 50,
            "acceptance_rate_mean": 0.375,
            "execution_elapsed_seconds": 7.0,
        })
        diagnostics = {"metrics": {"ess_bulk": 25.0, "ess_tail": 10.0},
                       "availability": {"ess_bulk": None, "ess_tail": None}}
        efficiency = summarize_efficiency(diagnostics, costs)
        self.assertEqual(efficiency["metrics"], {
            "ess_bulk_per_sampler_second": 5.0,
            "ess_bulk_per_likelihood_evaluation": 0.5,
            "ess_tail_per_sampler_second": 2.0,
            "ess_tail_per_likelihood_evaluation": 0.2,
        })
        json.dumps({"costs": costs, "efficiency": efficiency}, allow_nan=False)

    def test_missing_invalid_and_zero_costs_make_efficiency_unavailable(self):
        missing = summarize_sampler_costs(None, (0,))
        self.assertEqual(missing["availability"]["sampler_wall_seconds"], "missing-cost-records")
        zero = summarize_sampler_costs(
            [{"wall_seconds": 0.0, "log_likelihood_evaluations": 0, "acceptance_rate": 0.0}], (0,))
        diagnostics = {"metrics": {"ess_bulk": 4.0, "ess_tail": None},
                       "availability": {"ess_bulk": None, "ess_tail": "too-short"}}
        efficiency = summarize_efficiency(diagnostics, zero)
        self.assertEqual(efficiency["availability"]["ess_bulk_per_sampler_second"], "zero-denominator")
        self.assertEqual(efficiency["availability"]["ess_tail_per_likelihood_evaluation"], "too-short")

    def test_unavailable_comparison_inputs_are_not_subtracted(self):
        summary = summarize_selection(select_chain_array(np.full((2, 5), np.inf)))
        report = {"execution_id": "baseline", "samplers": {"sampler_0": {"variables": {"x": summary}}}}
        comparison = _compare_reports([report, {**report, "execution_id": "candidate"}])
        component = comparison["comparisons"]["candidate"]["sampler_0"]["variables"]["x"]["components"]["scalar"]
        self.assertIsNone(component["mean_delta"])
        self.assertEqual(component["availability"]["mean_delta"], "unavailable-input")
        json.dumps(comparison, allow_nan=False)


class BundleSelectionTests(unittest.TestCase):
    def test_exact_stored_and_unstored_warmup_bundle_selection_preserves_bytes(self):
        spec = polar_spec(chains=2, draws=3, warmup=2)
        source = np.arange(20, dtype=float).reshape(2, 5, 2)
        for stored in (True, False):
            with self.subTest(stored=stored), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "artifacts").mkdir()
                values = source if stored else source[:, 2:]
                archive = root / "artifacts/sampler_0.npz"
                np.savez(archive, x=values)
                (root / "spec.json").write_text(spec.to_json())
                manifest = {
                    "experiment": spec.to_dict(), "resolved_spec_sha256": content_hash(spec),
                    "record": {"execution_id": "fixture", "status": "completed", "run_id": spec.run_id,
                               "data_id": spec.data_id, "run_root": "/unavailable/original"},
                    "artifacts": [{"path": "artifacts/sampler_0.npz", "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                                   "size_bytes": archive.stat().st_size, "kind": "posterior"}],
                    "result_schemas": {"sampler_0": {"warmup_draws": 2, "warmup_included": stored,
                        "variables": [{"name": "x", "dims": ["chain", "draw", "parameter"], "shape": list(values.shape)}]}},
                }
                (root / "manifest.json").write_text(json.dumps(manifest))
                before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
                provenance = bundle_analysis_provenance(root)
                self.assertEqual(provenance["manifest_sha256"], hashlib.sha256(before[root / "manifest.json"]).hexdigest())
                self.assertEqual(provenance["artifacts"], manifest["artifacts"])
                self.assertEqual(provenance["manifest"], manifest)
                for cutoff in (None, 0, 2):
                    report = bundle_report_data(root / "manifest.json", cutoff=cutoff, chains=(1,))
                    applied = (2 if stored else 0) if cutoff is None else cutoff
                    variable = report["samplers"]["sampler_0"]["variables"]["x"]
                    self.assertEqual(variable["components"]["[0]"]["mean"], float(np.mean(values[1, applied:, 0])))
                    self.assertEqual(variable["selection"]["cutoff"], applied)
                    self.assertEqual(variable["selection"]["recorded_warmup_draws"], 2)
                    self.assertEqual(variable["selection"]["chain_ids"], [1])
                    json.dumps(report, allow_nan=False)
                self.assertEqual(before, {path: path.read_bytes() for path in root.rglob("*") if path.is_file()})


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
