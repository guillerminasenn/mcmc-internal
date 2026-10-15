"""Registry, run-root, sampler-seed, and detached-worker checks for the control plane."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from multiproposal.platform import (
    ExperimentSpec, ProblemSpec, RunStatus, SamplerSpec, SeedSpec,
)
from multiproposal.platform.registry import Registry, TransitionError, utc_now
from multiproposal.platform.runroot import RunRootError, resolve_run_root

CAPS = ("gaussian_prior", "log_likelihood", "log_density", "log_prior", "sample_prior")


def polar_spec(chains=2, draws=20, warmup=5, delay=None, rho=0.9):
    execution = {} if delay is None else {"chain_delay_seconds": delay}
    return ExperimentSpec(
        problem=ProblemSpec(
            "polar_twist", "polar-twist", "1.0",
            model={"alpha": 2.0, "sigma_noise": 0.1, "prior_mean": [0.0, 0.0],
                   "prior_cov": [[1.0, 0.0], [0.0, 1.0]]},
            data={"y_obs": [0.2, -0.1]}, capabilities=CAPS),
        samplers=(SamplerSpec("pcn", {"rho": rho}, ("gaussian_prior", "log_likelihood")),),
        seeds=SeedSpec(202, 0), chains=chains, draws=draws, warmup=warmup,
        execution=execution)


def provider_installed():
    from multiproposal.platform.validation import discover_providers
    return "polar-twist" in discover_providers()[0]


def run_cli(*args, root=None):
    command = [sys.executable, "-B", "-m", "multiproposal.platform.cli"]
    if root:
        command += ["--run-root", str(root)]
    result = subprocess.run(command + list(args), capture_output=True, text=True)
    return result


def wait_for(registry, execution_id, states, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = registry.reconcile(execution_id)
        if record.status in states:
            return record
        time.sleep(0.1)
    raise AssertionError(f"timed out in {registry.snapshot(execution_id).status}")


class RunRootTests(unittest.TestCase):
    def test_no_default_and_unignored_repository_rejected(self):
        os.environ.pop("MCMC_PLATFORM_RUN_ROOT", None)
        with self.assertRaisesRegex(RunRootError, "required"):
            resolve_run_root(None)
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["git", "init", "-q", directory], check=True)
            inside = Path(directory) / "runs"
            with self.assertRaisesRegex(RunRootError, "not ignored"):
                resolve_run_root(inside)
            self.assertEqual(resolve_run_root(inside, allow_repository=True), inside.resolve())
            (Path(directory) / ".gitignore").write_text("/runs/\n")
            self.assertEqual(resolve_run_root(inside), inside.resolve())
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(resolve_run_root(directory), Path(directory).resolve())


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.registry = Registry(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_registration_persists_spec_before_any_worker(self):
        spec = polar_spec()
        record = self.registry.register(spec)
        self.assertEqual(record.status, RunStatus.PENDING)
        self.assertEqual(self.registry.spec_text(record.execution_id), spec.to_json())
        self.assertEqual(Path(record.resolved_spec_path).read_text(), spec.to_json())
        retry = self.registry.register(spec)
        self.assertEqual(retry.run_id, record.run_id)
        self.assertNotEqual(retry.execution_id, record.execution_id)

    def test_spec_tampering_is_detected(self):
        record = self.registry.register(polar_spec())
        path = Path(record.resolved_spec_path)
        path.chmod(0o644)
        path.write_text(polar_spec(draws=21).to_json())
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.registry.spec_text(record.execution_id)

    def test_legal_and_illegal_transitions(self):
        record = self.registry.register(polar_spec())
        eid = record.execution_id
        self.assertTrue(self.registry.begin_running(eid, {"pid": 1}))
        self.assertFalse(self.registry.begin_running(eid, {"pid": 2}))
        self.registry.transition(eid, RunStatus.FAILED, error="boom")
        done = self.registry.snapshot(eid)
        self.assertEqual((done.status, done.error), (RunStatus.FAILED, "boom"))
        self.assertIsNotNone(done.finished_at)
        with self.assertRaises(TransitionError):
            self.registry.transition(eid, RunStatus.RUNNING)

    def test_dead_worker_becomes_stale_not_running(self):
        eid = self.registry.register(polar_spec()).execution_id
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        child.wait()
        from multiproposal.platform.registry import hostname
        self.registry.begin_running(
            eid, {"pid": child.pid, "host": hostname(), "start_ticks": None})
        record = self.registry.reconcile(eid)
        self.assertEqual(record.status, RunStatus.STALE)
        self.assertIn("not alive", record.error)

    def test_pending_without_worker_cancels_directly(self):
        from multiproposal.platform.cli import cancel
        eid = self.registry.register(polar_spec()).execution_id
        self.assertEqual(cancel(self.registry, eid).status, RunStatus.CANCELLED)


class SamplerSeedTests(unittest.TestCase):
    def setUp(self):
        try:
            import numpy as np
            from multiproposal.problems.base import GaussianPriorProblem
        except ImportError as exc:
            self.skipTest(str(exc))
        self.np = np

        class Problem(GaussianPriorProblem):
            def log_likelihood(self, x):
                return -float(np.dot(x - 1.0, x - 1.0))

        self.problem = Problem(np.zeros(2), np.eye(2))

    def run_chain(self, spec, chain):
        from multiproposal.platform.samplers import PCNSampler
        return PCNSampler().run(self.problem, spec, 0, chain)

    def test_seed_tree_reproducible_and_independent_of_order(self):
        spec = polar_spec(chains=3)
        first = [self.run_chain(spec, c) for c in range(3)]
        second = [self.run_chain(spec, c) for c in (2, 1, 0)][::-1]
        for a, b in zip(first, second):
            self.np.testing.assert_array_equal(a["arrays"]["parameters"],
                                               b["arrays"]["parameters"])
        self.assertFalse(self.np.array_equal(first[0]["arrays"]["parameters"],
                                             first[1]["arrays"]["parameters"]))
        other = polar_spec(chains=3).__class__(
            **{**{f: getattr(spec, f) for f in ("problem", "samplers", "chains", "draws", "warmup")},
               "seeds": SeedSpec(202, 1)})
        self.assertFalse(self.np.array_equal(
            first[0]["arrays"]["parameters"], self.run_chain(other, 0)["arrays"]["parameters"]))

    def test_shape_and_cost_counts(self):
        spec = polar_spec(draws=20, warmup=5)
        result = self.run_chain(spec, 0)
        self.assertEqual(result["arrays"]["parameters"].shape, (25, 2))
        self.assertEqual(result["costs"]["log_likelihood_evaluations"], 2 * 25)


@unittest.skipUnless(provider_installed(), "polar-twist provider is not installed")
class DetachedExecutionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name) / "runs"
        self.spec_path = Path(self.directory.name) / "spec.json"

    def tearDown(self):
        for record in Registry(self.root).list() if self.root.exists() else []:
            pid = record.worker.get("pid")
            if pid and record.status not in (RunStatus.COMPLETED, RunStatus.FAILED,
                                             RunStatus.CANCELLED):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.directory.cleanup()

    def submit(self, spec):
        self.spec_path.write_text(spec.to_json())
        result = run_cli("submit", str(self.spec_path), root=self.root)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)["execution_id"]

    def test_completion_is_observed_by_a_new_process_with_sealed_artifacts(self):
        eid = self.submit(polar_spec(chains=4, draws=30, warmup=10))
        registry = Registry(self.root)
        record = wait_for(registry, eid, {RunStatus.COMPLETED, RunStatus.FAILED})
        self.assertEqual(record.status, RunStatus.COMPLETED, record.error)
        status = json.loads(run_cli("status", eid, root=self.root).stdout)
        self.assertEqual(status["status"], "completed")
        directory = Path(record.run_root)
        manifest = json.loads((directory / "manifest.json").read_text())
        import hashlib
        for artifact in manifest["artifacts"]:
            data = (directory / artifact["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), artifact["sha256"])
            self.assertFalse(os.access(directory / artifact["path"], os.W_OK))
        import numpy as np
        schema = manifest["result_schemas"]["sampler_0"]
        variable_name = schema["variables"][0]["name"]
        artifact = next(item for item in manifest["artifacts"] if item["path"] == "artifacts/sampler_0.npz")
        dimensions = {
            variable["name"]: tuple(variable["dims"]) for variable in schema["variables"]
        }
        try:
            from bayeslab.results import read_npz_result
        except ImportError:
            self.skipTest("bayeslab is not installed")
        loaded = read_npz_result(
            directory / artifact["path"],
            sha256=artifact["sha256"],
            dimensions=dimensions,
            warmup=schema["warmup_draws"],
            warmup_stored=schema["warmup_included"],
        )
        self.assertEqual(set(loaded.arrays), {variable_name})
        self.assertEqual(loaded.arrays[variable_name].shape, (4, 40, 2))
        self.assertFalse(loaded.arrays[variable_name].flags.writeable)
        np.testing.assert_array_equal(
            loaded.select(variable_name, phase="posterior"),
            loaded.arrays[variable_name][:, 10:],
        )
        self.assertIn("starting", (directory / "logs" / "stdout.log").read_text())

    def test_cancel_running_worker(self):
        eid = self.submit(polar_spec(chains=2, delay=60))
        registry = Registry(self.root)
        wait_for(registry, eid, {RunStatus.RUNNING})
        cancelled = json.loads(run_cli("cancel", eid, root=self.root).stdout)
        self.assertIn(cancelled["status"], ("cancelling", "cancelled"))
        self.assertEqual(wait_for(registry, eid, {RunStatus.CANCELLED}).status,
                         RunStatus.CANCELLED)

    def test_killed_worker_is_reported_stale_with_logs_kept(self):
        eid = self.submit(polar_spec(chains=2, delay=60))
        registry = Registry(self.root)
        record = wait_for(registry, eid, {RunStatus.RUNNING})
        os.kill(record.worker["pid"], signal.SIGKILL)
        record = wait_for(registry, eid, {RunStatus.STALE})
        self.assertEqual(record.status, RunStatus.STALE)
        self.assertTrue(Path(record.logs["stdout"]).exists())

    def test_invalid_spec_fails_before_registration(self):
        bad = polar_spec(rho=1.5)
        self.spec_path.write_text(bad.to_json())
        result = run_cli("submit", str(self.spec_path), root=self.root)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(Registry(self.root).list(), [])

    def test_validate_and_dry_run_write_nothing(self):
        self.spec_path.write_text(polar_spec().to_json())
        self.assertEqual(run_cli("validate", str(self.spec_path)).returncode, 0)
        plan = json.loads(run_cli("dry-run", str(self.spec_path)).stdout)
        self.assertFalse(plan["writes_files"])
        self.assertEqual(plan["planned_log_likelihood_evaluations"]["sampler_0"], 2 * 25 * 2)
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
