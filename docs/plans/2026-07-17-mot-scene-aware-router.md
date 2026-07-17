# MoT Scene-Aware Router Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an opt-in, backward-compatible scene-aware routing residual and scene/expert consistency loss for the experimental MoT path.

**Architecture:** Preserve the existing token router and add a zero-initialized image-level residual derived from differentiable scene statistics. The residual broadcasts over spatial tokens, so local routing remains available while every token receives a scene prior; an optional consistency loss aligns high-frequency, spatially heterogeneous, and multi-scale scenes with the intended Local, Deformable, and Window experts.

**Tech Stack:** Python 3.11, PyTorch, pytest, YOLO model YAML

---

### Task 1: Define scene-aware router contracts

**Files:**
- Create: `tests/test_mot_scene_aware_router.py`

**Step 1: Test scene-statistic shape and finiteness**

Feed rectangular feature maps to the router and assert its scene-stat vector is `[B, 3]`, finite, and differentiable.

**Step 2: Test zero-init backward compatibility**

Create legacy and scene-aware routers with identical legacy router weights. Assert their outputs match exactly before the scene projection learns.

**Step 3: Test scene residual can change routing**

Set a deterministic scene projection and assert smooth and high-frequency inputs produce different expert weights.

**Step 4: Test scene consistency loss**

Construct scene stats and matching/mismatching expert weights. Assert matching routing has lower loss and gradients reach the scene projection.

**Step 5: Test wrapper and config plumbing**

Assert `C2fMoT(..., scene_aware_router=True)` configures every child block, and the new experimental YAML parses and runs a minimal forward.

**Step 6: Run the tests to verify they fail**

Run: `python -m pytest -q tests/test_mot_scene_aware_router.py --tb=long`

Expected: FAIL because scene-aware options are not implemented.

### Task 2: Implement the opt-in scene residual

**Files:**
- Modify: `ultralytics/nn/modules/mot/router.py`

**Step 1: Add differentiable scene statistics**

Compute normalized high-frequency energy, spatial heterogeneity, and multi-scale variance from the input feature map in float32.

**Step 2: Add zero-initialized scene projection**

When enabled, map the three statistics through a small MLP to expert logits and add the result to legacy router logits. Zero-initialize the final linear layer for exact initial parity.

**Step 3: Expose diagnostics**

Store detached `last_scene_stats` and `last_scene_bias` for routing analysis scripts without adding persistent buffers.

**Step 4: Add scene consistency loss**

Build a soft target distribution with Local driven by high-frequency energy, Window by multi-scale variance, and Deformable by spatial heterogeneity. Compare mean router probabilities with the target using KL divergence.

### Task 3: Integrate blocks, wrappers, and config

**Files:**
- Modify: `ultralytics/nn/modules/mot/block.py`
- Modify: `ultralytics/nn/modules/mot/wrappers.py`
- Modify: `ultralytics/nn/modules/moe/config.py`
- Modify: `ultralytics/cfg/default.yaml`

**Step 1: Extend `MoTBlock` and `C2fMoT`**

Add `scene_aware_router=False`, `scene_hidden_dim=None`, and `scene_consistency_coeff=0.0`, preserving all existing positional arguments.

**Step 2: Add consistency loss to aux loss**

Only compute it during training when the coefficient is positive. Record the component and scene stats in the routing snapshot.

**Step 3: Add runtime config fields**

Expose `mot_scene_aware_router`, `mot_scene_hidden_dim`, and `mot_scene_consistency` through `default.yaml`, `MIXTURE_DEFAULTS`, `CLI_FIELDS`, and `apply_mixture_config`.

### Task 4: Add an experimental architecture config

**Files:**
- Create: `ultralytics/cfg/models/master/v0_10/det/yolo-master-mot-scene-n.yaml`
- Modify: `docs/governance/model-registry.yaml`

**Step 1: Copy the v0.10 MoT architecture**

Enable scene-aware routing and a conservative consistency coefficient in each `C2fMoT` layer without changing the existing v0.10 config.

**Step 2: Register it as experimental**

Record parse, forward, and unit-test evidence only. Do not mark mAP improvement or deployment verification before experiments.

### Task 5: Verify implementation

**Files:**
- Test: `tests/test_mot_scene_aware_router.py`
- Test: `tests/test_mot.py`
- Test: `tests/test_mixture_config_resolution.py`
- Test: `tests/test_mixture_compile.py`
- Test: `tests/test_mixture_export.py`

**Step 1: Run focused and MoT tests**

Run: `python -m pytest -q tests/test_mot_scene_aware_router.py tests/test_mot.py tests/test_mixture_config_resolution.py --tb=long`

Expected: PASS.

**Step 2: Run compile and export checks**

Run: `python -m pytest -q tests/test_mixture_compile.py tests/test_mixture_export.py -k mot --tb=long`

Expected: PASS or dependency skips only.

**Step 3: Run configuration and lint gates**

Run: `python tools/config_drift_detector.py`

Run: `ruff check ultralytics/nn/modules/mot tests/test_mot_scene_aware_router.py`

Expected: PASS.

### Task 6: Prepare the experiment handoff

**Files:**
- Modify: `docs/mot_integration_experiment_report_2026-06-25.md`

**Step 1: Add the new ablation row**

Define baseline v0.10 MoT versus scene-aware MoT, fixed seed/data/hyperparameters, routing diagnostics, mAP50-95, APs/m/l, latency, and memory.

**Step 2: State the acceptance rule**

MoT remains experimental until a 50-epoch VisDrone run reaches at least parity with the MoE baseline and shows increased Deformable activation on irregular/occluded subsets.
