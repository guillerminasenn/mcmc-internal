"""Dependency-free checks for platform contracts, without running chains."""

from dataclasses import FrozenInstanceError, replace
import json
import unittest

from multiproposal.platform import (
    ArtifactRecord,
    ExecutionClient,
    ExperimentSpec,
    ForwardProblem,
    GaussianPriorAdapter,
    GaussianPriorRuntime,
    ProblemProvider,
    ProblemSpec,
    ProviderMetadata,
    ResultSchema,
    RunManifest,
    RunRecord,
    RunStatus,
    RuntimeProblem,
    SamplerSpec,
    SeedSpec,
    VariableSpec,
    canonical_json,
    content_hash,
    validate_runtime,
)


def example_spec(**overrides):
    """Build a small resolved fixture, not a runnable provider or sampler."""
    values = {
        "problem": ProblemSpec(
            problem_id="fixture", provider="fixture-provider", provider_version="1",
            model={"prior_mean": [0.0, 0.0]}, data={"observations": [1.0, 2.0]},
            capabilities=("log_likelihood", "sample_prior", "gaussian_prior"),
        ),
        "samplers": (SamplerSpec(
            sampler_id="fixture-sampler", config={"M": 4},
            required_capabilities=("log_likelihood", "sample_prior", "gaussian_prior"),
        ),),
        "seeds": SeedSpec(data_seed=202, experiment_seed=0),
        "chains": 4,
        "draws": 20,
        "warmup": 5,
    }
    values.update(overrides)
    return ExperimentSpec(**values)


class SpecificationTests(unittest.TestCase):
    """Check serialization, identity, compatibility, and seed ownership."""

    def test_canonical_json_and_sha256_golden(self):
        """Freeze the encoding and SHA-256 convention independently."""
        self.assertEqual(canonical_json({"b": [2, True], "a": 1}), '{"a":1,"b":[2,true]}')
        self.assertEqual(
            content_hash({}),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        )

    def test_json_rejects_nonfinite_and_non_json_values(self):
        """Reject values that cannot be portable scientific configuration."""
        for value in (float("nan"), float("inf"), {1: "bad"}, object(), {1, 2}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                canonical_json(value)

    def test_round_trip(self):
        """CLI and UI can persist and reload identical resolved bytes."""
        original = example_spec()
        restored = ExperimentSpec.from_json(original.to_json())
        self.assertEqual(restored.to_json(), original.to_json())
        self.assertEqual(restored.run_id, original.run_id)
        self.assertEqual(restored.data_id, original.data_id)

    def test_execution_options_do_not_change_scientific_ids(self):
        """Worker placement is persisted but does not redefine the experiment."""
        original = example_spec()
        rescheduled = replace(original, execution={"workers": 8, "output_root": "/tmp/example"})
        self.assertEqual(original.run_id, rescheduled.run_id)
        self.assertEqual(original.data_id, rescheduled.data_id)
        self.assertNotEqual(original.to_json(), rescheduled.to_json())

    def test_statistical_changes_change_run_identity(self):
        """Counts, sampling entropy, and algorithm settings cannot reuse a run ID."""
        original = example_spec()
        variants = (
            replace(original, draws=21), replace(original, warmup=6),
            replace(original, chains=5),
            replace(original, seeds=SeedSpec(data_seed=202, experiment_seed=1)),
            replace(original, samplers=(replace(original.samplers[0], config={"M": 5}),)),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertNotEqual(original.run_id, variant.run_id)
                self.assertEqual(original.data_id, variant.data_id)

    def test_data_seed_and_observations_change_data_identity(self):
        """Model/data settings and data entropy participate in data identity."""
        original = example_spec()
        variants = (
            replace(original, seeds=SeedSpec(data_seed=203, experiment_seed=0)),
            replace(original, problem=replace(original.problem, data={"observations": [3.0, 4.0]})),
            replace(original, problem=replace(original.problem, model={"prior_mean": [1.0, 0.0]})),
        )
        for variant in variants:
            self.assertNotEqual(original.data_id, variant.data_id)
            self.assertNotEqual(original.run_id, variant.run_id)

    def test_configuration_is_deeply_detached_and_immutable(self):
        """A caller cannot mutate a spec or its ID through nested configuration."""
        model = {"prior": {"mean": [0.0, 0.0]}}
        problem = replace(example_spec().problem, model=model)
        original_json = canonical_json(problem)
        model["prior"]["mean"][0] = 99.0
        problem.to_dict()["model"]["prior"]["mean"][0] = 100.0
        self.assertEqual(canonical_json(problem), original_json)
        with self.assertRaises(TypeError):
            problem.model["other"] = 0
        with self.assertRaises(TypeError):
            problem.model["prior"]["mean"][0] = 1
        with self.assertRaises(FrozenInstanceError):
            problem.problem_id = "changed"

    def test_capabilities_are_order_independent(self):
        """Equivalent capability declarations have the same identity."""
        original = example_spec()
        reordered = replace(original.problem, capabilities=tuple(reversed(original.problem.capabilities)))
        self.assertEqual(replace(original, problem=reordered).run_id, original.run_id)

    def test_missing_capability_is_rejected(self):
        """Reject incompatible problem/sampler pairs before execution."""
        with self.assertRaisesRegex(ValueError, "missing capabilities.*gradient"):
            example_spec(samplers=(SamplerSpec("hmc-fixture", required_capabilities=("gradient",)),))

    def test_unknown_versions_and_fields_are_rejected(self):
        """Do not silently accept incompatible schemas or discard fields."""
        for field_name, value in (("schema_version", 2), ("hash_version", "future"), ("unknown", 1)):
            payload = example_spec().to_dict()
            payload[field_name] = value
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                ExperimentSpec.from_json(json.dumps(payload))

    def test_invalid_counts_and_seeds_are_rejected(self):
        """Counts and entropy must be meaningful integers, not booleans."""
        for field_name, value in (("chains", 0), ("draws", -1), ("warmup", -1), ("chains", True)):
            with self.subTest(field_name=field_name), self.assertRaises(ValueError):
                example_spec(**{field_name: value})
        with self.assertRaises(ValueError):
            SeedSpec(data_seed=-1, experiment_seed=0)

    def test_unversioned_or_ambiguous_json_is_rejected(self):
        """Do not silently reinterpret unversioned files or duplicate field values."""
        for missing in ("schema_version", "hash_version"):
            payload = example_spec().to_dict()
            payload.pop(missing)
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                ExperimentSpec.from_json(json.dumps(payload))
        payload = example_spec().to_dict()
        payload["problem"].pop("schema_version")
        with self.assertRaises(ValueError):
            ExperimentSpec.from_json(json.dumps(payload))
        text = example_spec().to_json().replace('"chains":4', '"chains":4,"chains":8')
        with self.assertRaisesRegex(ValueError, "Duplicate JSON field"):
            ExperimentSpec.from_json(text)

    def test_seed_paths_follow_chain_and_sampler_not_shards(self):
        """Describe distinct task-owned streams without importing NumPy."""
        seeds = SeedSpec(data_seed=202, experiment_seed=0)
        self.assertEqual(seeds.sampler_spawn_key(2, 1), (0, 2, 1))
        self.assertNotEqual(seeds.sampler_spawn_key(2, 1), seeds.sampler_spawn_key(1, 2))
        with self.assertRaises(ValueError):
            seeds.sampler_spawn_key(-1, 0)


class ProviderTests(unittest.TestCase):
    """Exercise structural providers, labeled dimensions, and legacy delegation."""

    def test_provider_implements_no_platform_base_class(self):
        """Ordinary Python classes and dictionaries satisfy the provider boundary."""
        class PlainProvider:
            def describe(self):
                return {
                    "schema_version": 1,
                    "problem_id": "fixture", "label": "Fixture", "provider_version": "1",
                    "parameters": [{"name": "field", "dims": ["row", "column"],
                                    "shape": [3, 5], "coordinates": {"row": [0, 1, 2]}}],
                    "capabilities": [],
                }

            def build(self, model, data):
                return {"model": model, "data": data}

        provider = PlainProvider()
        self.assertIsInstance(provider, ProblemProvider)
        self.assertNotIsInstance(provider, ForwardProblem)
        metadata = ProviderMetadata.from_dict(provider.describe())
        self.assertEqual(metadata.parameters[0].shape, (3, 5))
        self.assertEqual(metadata.parameters[0].dims, ("row", "column"))
        self.assertEqual(ProviderMetadata.from_dict(metadata.to_dict()), metadata)
        self.assertEqual(provider.build({"setting": 1}, {"observed": 2})["data"], {"observed": 2})

    def test_variable_coordinate_lengths_and_dimensions_are_checked(self):
        """Do not mislabel matrix axes or silently broadcast coordinates."""
        for coordinates in ({"unknown": [0, 1]}, {"row": [0]}, {"row": [[0], [1]]}):
            with self.subTest(coordinates=coordinates), self.assertRaises(ValueError):
                VariableSpec("field", dims=("row",), shape=(2,), coordinates=coordinates)
        with self.assertRaises(ValueError):
            VariableSpec("field", dims=("row", "row"), shape=(2, 2))
        with self.assertRaises(ValueError):
            VariableSpec("field", dims=("row",), shape=(True,))

    def test_metadata_rejects_duplicate_variables_and_unknown_schema(self):
        """Provider metadata has explicit variables and a supported version."""
        variable = VariableSpec("x")
        with self.assertRaises(ValueError):
            ProviderMetadata("fixture", "Fixture", "1", parameters=(variable, variable))
        with self.assertRaises(ValueError):
            ProviderMetadata.from_dict({"schema_version": 2})

    def test_declared_capabilities_require_runtime_operations(self):
        """A metadata claim is not enough to satisfy a known sampler operation."""
        class DensityOnly:
            dim = 2

            def log_posterior(self, parameters):
                return -sum(value * value for value in parameters)

        target = DensityOnly()
        self.assertIsInstance(target, RuntimeProblem)
        validate_runtime(target, ("log_density",))
        with self.assertRaisesRegex(ValueError, "sample_prior"):
            validate_runtime(target, ("sample_prior",))
        with self.assertRaises(ValueError):
            GaussianPriorAdapter(target)

    def test_adapter_preserves_real_gaussian_problem_values_and_rng(self):
        """Compare real legacy prior draws and densities without running MCMC."""
        try:
            import numpy as np
            from multiproposal.problems.base import GaussianPriorProblem
        except ImportError as exc:
            self.skipTest(f"Numerical adapter test requires NumPy/SciPy: {exc}")

        class LikelihoodProblem(GaussianPriorProblem):
            def log_likelihood(self, parameters):
                return -float(np.dot(parameters, parameters))

        problem = LikelihoodProblem(np.array([1.0, -2.0]), np.array([[2.0, 0.2], [0.2, 1.0]]))
        adapter = GaussianPriorAdapter(problem)
        self.assertIsInstance(adapter, GaussianPriorRuntime)
        parameters = np.array([0.5, 1.0])
        self.assertEqual(adapter.dim, problem.dim)
        self.assertIs(adapter.prior_mean(), problem.prior_mean())
        self.assertEqual(adapter.log_prior(parameters), problem.log_prior(parameters))
        self.assertEqual(adapter.log_likelihood(parameters), problem.log_likelihood(parameters))
        self.assertEqual(adapter.log_posterior(parameters), problem.log_posterior(parameters))
        direct_rng = np.random.default_rng(42)
        adapted_rng = np.random.default_rng(42)
        np.testing.assert_array_equal(problem.sample_prior(direct_rng), adapter.sample_prior(adapted_rng))
        self.assertEqual(direct_rng.bit_generator.state, adapted_rng.bit_generator.state)


class ResultAndExecutionTests(unittest.TestCase):
    """Validate result labeling and truthful execution/provenance snapshots."""

    def pending_record(self, **overrides):
        """Create a registry snapshot without launching or registering a process."""
        spec = example_spec()
        values = {
            "execution_id": "fixture-attempt-1", "run_id": spec.run_id,
            "data_id": spec.data_id, "status": RunStatus.PENDING,
            "created_at": "2026-10-14T12:00:00+00:00",
            "resolved_spec_path": "/configured/runs/fixture/spec.json",
            "run_root": "/configured/runs/fixture",
        }
        values.update(overrides)
        return RunRecord(**values)

    def test_artifact_paths_and_checksums_are_explicit(self):
        """Reject traversal, ambiguous path syntax, and unsupported checksums."""
        artifact = ArtifactRecord("samples/chain.npz", "a" * 64, 100, "posterior")
        self.assertEqual(artifact.to_dict()["sha256"], "a" * 64)
        for path in ("../chain.npz", "/chain.npz", "a/../chain.npz", "C:/chain.npz", ".", "a//b", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(artifact, path=path)
        with self.assertRaises(ValueError):
            replace(artifact, sha256="a" * 40)
        with self.assertRaises(ValueError):
            replace(artifact, size_bytes=-1)

    def test_chain_draw_and_warmup_semantics(self):
        """Require explicit sampling axes and leave at least one posterior draw."""
        variable = VariableSpec("x", dims=("chain", "draw", "parameter"), shape=(4, 20, 2))
        schema = ResultSchema((variable,), warmup_draws=5, warmup_included=False)
        self.assertEqual(schema.to_dict()["variables"][0]["dims"], ["chain", "draw", "parameter"])
        with self.assertRaises(ValueError):
            ResultSchema((replace(variable, dims=("draw", "chain", "parameter")),))
        with self.assertRaises(ValueError):
            ResultSchema((variable,), warmup_draws=20, warmup_included=True)
        with self.assertRaises(ValueError):
            ResultSchema((variable, replace(variable, name="y", shape=(3, 20, 2))))

    def test_array_shape_validation_does_not_mutate_data(self):
        """Schema checks need only array shape metadata, not numerical imports."""
        class ReadOnlyFixture:
            shape = (4, 20, 2)

        array = ReadOnlyFixture()
        schema = ResultSchema((VariableSpec("x", dims=("chain", "draw", "parameter"), shape=array.shape),))
        schema.validate_arrays({"x": array})
        self.assertEqual(array.shape, (4, 20, 2))
        with self.assertRaises(ValueError):
            schema.validate_arrays({"missing": array})
        with self.assertRaises(ValueError):
            replace(schema, variables=(replace(schema.variables[0], shape=(4, 20, 3)),)).validate_arrays({"x": array})

    def test_status_and_timestamp_validation(self):
        """Do not represent a pending or uncertain worker as completed."""
        record = self.pending_record()
        self.assertEqual(record.status, RunStatus.PENDING)
        self.assertEqual(replace(record, status=RunStatus.STALE).status, RunStatus.STALE)
        for overrides in (
            {"status": "unknown-status"}, {"status": RunStatus.RUNNING},
            {"status": RunStatus.COMPLETED}, {"created_at": "2026-10-14T12:00:00"},
            {"status": RunStatus.FAILED, "finished_at": "2026-10-13T12:00:00+00:00"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                replace(record, **overrides)

    def test_manifest_identity_and_completed_evidence(self):
        """A new manifest cannot join unrelated specs or claim empty completion."""
        manifest = RunManifest(example_spec(), self.pending_record())
        self.assertEqual(manifest.resolved_spec_sha256, content_hash(example_spec()))
        self.assertEqual(manifest.to_dict()["resolved_spec_sha256"], content_hash(example_spec()))
        self.assertEqual(manifest.to_dict()["record"]["status"], "pending")
        with self.assertRaises(ValueError):
            replace(manifest, record=replace(manifest.record, run_id="another-run"))
        complete = replace(manifest.record, status=RunStatus.COMPLETED, finished_at="2026-10-14T12:01:00+00:00")
        with self.assertRaises(ValueError):
            replace(manifest, record=complete)
        artifact = ArtifactRecord("samples.npz", "a" * 64, 100, "posterior")
        schema = ResultSchema(
            (VariableSpec("x", dims=("chain", "draw", "parameter"), shape=(4, 20, 2)),),
            warmup_draws=5,
        )
        with self.assertRaisesRegex(ValueError, "provenance"):
            replace(manifest, record=complete, artifacts=(artifact,), result_schemas={"sampler_0": schema})
        sealed = replace(
            manifest, record=complete, artifacts=(artifact,), result_schemas={"sampler_0": schema},
            code_revisions={"platform": {"commit": "fixture", "dirty": False}},
            environment={"python": "3.12", "packages": {}},
        )
        self.assertEqual(sealed.to_dict()["result_schemas"]["sampler_0"]["variables"][0]["shape"], [4, 20, 2])
        self.assertEqual(sealed.to_dict()["environment_sha256"], content_hash(sealed.environment))
        for invalid in (
            replace(schema, warmup_draws=0),
            replace(schema, variables=(replace(schema.variables[0], shape=(3, 20, 2)),)),
            replace(schema, variables=(replace(schema.variables[0], shape=(4, 19, 2)),)),
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                replace(sealed, result_schemas={"sampler_0": invalid})

    def test_retry_identity_and_client_are_structural(self):
        """Retry placement changes execution identity, not the scientific spec."""
        record = self.pending_record()
        retry = replace(record, execution_id="fixture-attempt-2")
        self.assertEqual(record.run_id, retry.run_id)
        self.assertNotEqual(record.execution_id, retry.execution_id)

        class ClientFixture:
            def validate(self, spec):
                return None

            def dry_run(self, spec):
                return {"run_id": spec.run_id}

            def submit(self, spec):
                return record

            def status(self, execution_id):
                return record

            def cancel(self, execution_id):
                return record

        self.assertIsInstance(ClientFixture(), ExecutionClient)


if __name__ == "__main__":
    unittest.main()