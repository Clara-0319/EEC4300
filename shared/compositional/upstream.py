"""Read-only bridge to the preserved upstream runners, validators and canonicalizers."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT


@lru_cache(None)
def load(relative):
    path = UPSTREAM / relative
    name = "hypo_preserved_" + relative.replace("/", "_").replace(".", "_")
    # Original runners use absolute 'modules' imports. Those templates are identical
    # in this pinned snapshot. Keep all writes (including pyc) out of the snapshot.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class BooleanAdapter:
    def __init__(self, metadata, observations):
        self.metadata = metadata
        self.public_observations = observations
        self.data = load("boolean/boolean_dataset.py")
        runner = load("boolean/boolean_benchmark.py")
        self.runner = runner.BooleanBenchmarkRefined.__new__(runner.BooleanBenchmarkRefined)
        self.runner.metadata = metadata
        self.runner.complete_dataset = {"metadata": metadata}
        self.runner.variables = metadata["variables"]
        self.runner.operators = set(metadata["operators"])
        self.runner.max_depth = metadata["max_depth"]
        self.runner.mechanistic_opts = metadata["mechanistic_opts"]
        self.observations = [self.data.BooleanObservation(o["inputs"], o["output"]) for o in observations]

    def evaluate(self, payload):
        # Guard the upstream validator's permissive parse-failure path. Only parsed,
        # in-space proposals may count. The preserved baseline itself is untouched.
        # sympify evaluates Python-like input; allow only the documented expression
        # alphabet before delegating to upstream (especially for live LLM output).
        lexical = isinstance(payload, str) and not re.sub(r"[A-Za-z_][A-Za-z_0-9]*|[\s(),]", "", payload)
        allowed = set(self.metadata["variables"]) | set(self.metadata["operators"])
        # NOR is serialized as NOT(OR(...)) to avoid the upstream nested-NOR parser.
        if "NOR" in allowed:
            allowed |= {"NOT", "OR"}
        lexical = lexical and set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", payload)) <= allowed
        in_space = bool(lexical and self.runner._in_space(payload))
        key = self.runner.get_expression_mechanistic_key(payload) if in_space else None
        valid = bool(in_space and self.runner.validate_expression(payload, self.observations)[0])
        return {"in_space": in_space, "valid": valid, "key": repr(key) if key is not None else None,
                "reason": "valid" if valid else "observation_mismatch" if in_space else "parse_or_space_error"}

    def ground_truth_keys(self, case):
        keys = set()
        for item in case["ground_truth_expressions"]:
            key = item.get("mechanistic_key")
            if key is not None:
                key = ast.literal_eval(key) if isinstance(key, str) else key
            else:
                key = self.runner.get_expression_mechanistic_key(item["formula"])
            if key is None:
                raise ValueError("Unparseable Boolean ground truth")
            keys.add(repr(key))
        return keys

    def prompt(self, history):
        return self.runner.create_prompt(self.observations, history)

    def parse(self, response):
        return self.runner.parse_llm_response(response)


class VoxelAdapter:
    def __init__(self, metadata, top):
        self.metadata = metadata
        self.public_observations = top
        self.data = load("3d/run_3d_benchmark.py")
        self.runner = self.data.Benchmark3D.__new__(self.data.Benchmark3D)
        self.runner.metadata = metadata
        self.top = top
        self.size = metadata["grid_size"]

    def canonical_key(self, obj):
        if any(layer.any() for layer in obj.layers):
            return obj.get_hash()
        # The preserved normalizer passes a NumPy array to a list-based constructor
        # for all-zero grids. Match its intended single-zero-layer hash safely here.
        import numpy as np
        arr = np.zeros((1, self.size, self.size), dtype=np.uint8)
        digest = hashlib.md5()
        digest.update(arr.tobytes())
        digest.update(np.array(arr.shape, dtype=np.int64).tobytes())
        return digest.hexdigest()[:8]

    def evaluate(self, payload):
        try:
            in_space = (isinstance(payload, list) and 0 < len(payload) <= self.metadata["max_height"]
                        and all(len(layer) == self.size and all(
                            len(row) == self.size and all(type(x) is int and x in (0, 1) for x in row)
                            for row in layer) for layer in payload))
            if not in_space:
                raise ValueError("Out of grid/height/binary space")
            obj = self.data.Structure3D(payload)
            key = self.canonical_key(obj)
            valid = self.runner.validate_structure_matches_observations(obj, self.top)
            return {"in_space": True, "valid": bool(valid), "key": key,
                    "reason": "valid" if valid else "projection_or_gravity_error"}
        except (TypeError, ValueError, IndexError):
            return {"in_space": False, "valid": False, "key": None, "reason": "parse_or_space_error"}

    def ground_truth_keys(self, case):
        keys = set()
        for gt in case["ground_truth_structures"]:
            layers = gt["layers"]
            if layers and isinstance(layers[0], str):
                layers = [[[int(x) for x in bits[i:i + self.size]]
                           for i in range(0, len(bits), self.size)] for bits in layers]
            keys.add(self.canonical_key(self.data.Structure3D(layers)))
        return keys

    def prompt(self, history):
        return self.runner.create_prompt(self.top, [self.data.Structure3D(h) for h in history if h])

    def parse(self, response):
        obj = self.runner.parse_llm_response(response)
        return [layer.tolist() for layer in obj.layers] if obj else None

