# Routing Interpreter Toolkit Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a unified, non-invasive routing interpretation toolkit for MoE, MoA, MoT, and MoLoRA models.

**Architecture:** Observe routed modules through the existing `last_routing_snapshot` contract and temporary router forward hooks. Keep the interpreter outside model state so diagnostics, forced-routing counterfactuals, and heatmap capture do not alter checkpoints, export behavior, or default forward paths.

**Tech Stack:** Python 3.11, PyTorch, matplotlib, pytest

---

### Task 1: Define cross-family interpretation contracts

**Files:**
- Create: `tests/test_routing_interpreter.py`

**Step 1: Test snapshot collection**

Build a small routed model and assert the interpreter discovers leaf routed layers, normalizes expert usage, and reports stable layer names and module types.

**Step 2: Test collapse detection**

Feed deliberately collapsed expert usage and assert dominant share, normalized Gini, entropy, dead experts, and the combined collapse flag are reported.

**Step 3: Test specialization analysis**

Run a small dataset through deterministic routing and assert per-expert activation, dominant-sample rate, and input-feature signatures are aggregated per layer.

**Step 4: Test spatial heatmap capture**

Capture a token router and assert probabilities use `[B, E, H, W]`, sum to one, and assignment maps preserve spatial dimensions.

**Step 5: Test forced-routing causal analysis**

Force two different experts without changing model parameters and assert output-distance metrics differ while training flags and router behavior are restored afterward.

**Step 6: Run tests to verify they fail**

Run: `python -m pytest -q tests/test_routing_interpreter.py --tb=long`

Expected: FAIL because `ultralytics.utils.routing_interpreter` does not exist.

### Task 2: Implement the interpreter core

**Files:**
- Create: `ultralytics/utils/routing_interpreter.py`

**Step 1: Add structured reports**

Define immutable JSON-compatible records for layer summaries, collapse diagnostics, specialization reports, heatmaps, and causal metrics.

**Step 2: Add routed-layer discovery and snapshot normalization**

Use the routed-module protocol, prefer leaf routed modules to avoid wrapper duplication, validate expert vector sizes, and normalize finite non-negative usage.

**Step 3: Add temporary router observation hooks**

Support tensor and tuple router outputs, infer the expert axis from `num_experts`, convert logits to probabilities when necessary, and preserve batch/spatial dimensions.

**Step 4: Add dataset specialization analysis**

Dispatch tensor, tuple/list, and dict batches through the model or an optional `forward_fn`. Aggregate mean usage, dominant routing, and either caller-provided or default tensor descriptors.

**Step 5: Add collapse detection**

Compute normalized Gini, normalized entropy, dominant share, and dead experts. Support both existing snapshots and dataset-aggregated usage.

**Step 6: Add heatmap rendering**

Return raw routing probability/assignment tensors and optionally write deterministic PNG panels with matplotlib.

**Step 7: Add forced-routing counterfactuals**

Temporarily replace router outputs with a selected expert, compare nested tensor outputs with the natural route, and restore all hooks and training flags in `finally`.

### Task 3: Add the command-line entry point and documentation

**Files:**
- Create: `tools/routing_interpreter.py`
- Create: `docs/governance/routing-interpretability.md`

**Step 1: Add a model/image CLI**

Load a YOLO checkpoint, run prediction on one image, capture a selected routed layer or all leaf routed layers, and write heatmap PNGs plus a JSON summary.

**Step 2: Document Python and CLI usage**

Document supported module families, batch dispatch rules, custom feature descriptors, causal-analysis limits, and the fact that forced routing is a diagnostic counterfactual rather than a deployment mode.

### Task 4: Verify the integration

**Files:**
- Test: `tests/test_routing_interpreter.py`
- Test: `tests/test_routed_module_protocol.py`
- Test: `tests/test_mot_scene_aware_router.py`
- Test: `tests/test_molora_routing_aware_merge.py`

**Step 1: Run focused tests**

Run: `python -m pytest -q tests/test_routing_interpreter.py --tb=long`

Expected: PASS.

**Step 2: Run routed-module regressions**

Run: `python -m pytest -q tests/test_routed_module_protocol.py tests/test_mot_scene_aware_router.py tests/test_molora_routing_aware_merge.py --tb=long`

Expected: PASS.

**Step 3: Run lint and compile checks**

Run: `ruff check ultralytics/utils/routing_interpreter.py tools/routing_interpreter.py tests/test_routing_interpreter.py`

Run: `python -m compileall -q ultralytics/utils/routing_interpreter.py tools/routing_interpreter.py`

Expected: PASS with no output.
