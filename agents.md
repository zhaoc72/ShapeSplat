# AGENTS.md — ShapeSplat++ Full Project Execution Guide

This file defines the mandatory execution, coding, experiment, and research rules for all agents working on this project.

The project has moved beyond minimal/debug scaffolding. Future work must focus on completing the full ShapeSplat++ project toward paper-level experiments and reproducible results.

---

## 1. Project Identity

Project name:

```text
ShapeSplat++
```

Working title:

```text
SAM3-DINOv3-guided Object-Centric Visible-Hidden Gaussian Reconstruction
```

Current project root:

```text
C:\Users\zhaoc\ShapeSplat
```

Important:

```text
The old nested project paths are deprecated:
C:\Users\zhaoc\ShapeSplat\shapesplat_minimal
C:\Users\zhaoc\ShapeSplat\shapesplat
All future code edits, commands, tests, outputs, and experiments must use:
C:\Users\zhaoc\ShapeSplat
```

The project goal is no longer a minimal demo. Do not keep expanding minimal/debug scaffolding unless it protects tests or is explicitly requested. The goal is a complete research codebase that can run:

```text
CO3Dv2 diagnostics
same-mask object reconstruction
Ours benchmark
Ours variants / ablations
editing evaluation
stress benchmark
baseline comparison
final paper-style experiments
```

---

## 2. Mandatory Runtime Environment

All commands, dependency installation, tests, code validation, and experiment runs must be executed inside the `shapesplat` conda environment.

Preferred shell:

```text
Anaconda Prompt
```

Do not use Windows PowerShell syntax unless explicitly requested.

Every command session must begin with:

```bat
conda activate shapesplat
cd /d C:\Users\zhaoc\ShapeSplat
```

Use Windows `cmd` / Anaconda Prompt compatible commands. Prefer one-line commands.

Do not use Bash heredoc syntax such as:

```bash
python - <<'PY'
```

Do not use PowerShell backtick line continuation in suggested commands.

---

## 3. Local Hardware and Runtime

Local machine:

```text
OS: Windows
CPU: Intel
GPU: NVIDIA GeForce RTX 5090
Conda env: shapesplat
```

PyTorch CUDA has been confirmed working after installing CUDA-enabled PyTorch in the `shapesplat` environment.

Known working CUDA configuration observed previously:

```text
torch: 2.8.0+cu128
torch cuda: 12.8
cuda available: true
GPU: NVIDIA GeForce RTX 5090
compute capability: [12, 0]
```

Before GPU experiments, always run:

```bat
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime_cuda
```

For paper-level experiments, do not silently fall back to CPU when CUDA is requested. CPU fallback is only allowed when explicitly enabled.

---

## 4. Local Data and Model Paths

### 4.1 CO3Dv2 single subset

Local dataset path:

```text
D:\projects\datasets\co3dv2_single
```

Converted benchmark path:

```text
data\co3dv2_single_benchmark\manifest.csv
```

CO3Dv2 single is an object-centric single-foreground dataset. It should be treated as:

```text
real-image diagnostic / single-object visible-mask benchmark
```

Do not describe CO3Dv2 single as a multi-object occlusion benchmark.

### 4.2 DINOv3

Official reference:

```text
https://github.com/facebookresearch/dinov3
```

Local repo path:

```text
D:\projects\dinov3
```

Local weights:

```text
D:\projects\dinov3\checkpoint\dinov3_vits16_pretrain_lvd1689m.pth
D:\projects\dinov3\checkpoint\dinov3_vitl16_pretrain_lvd1689m.pth
```

Usage policy:

```text
ViT-S/16: first debug and pipeline validation
ViT-L/16: main experiment default after ViT-S/16 succeeds
```

DINOv3 is used as a frozen dense feature extractor. It must output dense patch features:

```text
[D, H, W]
```

Do not use global image embeddings such as:

```text
[B, D]
```

as dense features for formal experiments.

Known DINOv3 dependency issue:

If DINOv3 loading fails with:

```text
No module named 'torchmetrics'
```

run:

```bat
python -m pip install torchmetrics omegaconf ftfy regex scikit-learn submitit termcolor
```

Do not blindly run:

```bat
python -m pip install -r D:\projects\dinov3\requirements.txt
```

if doing so may overwrite the working CUDA PyTorch installation.

### 4.3 SAM3

Official references:

```text
https://github.com/facebookresearch/sam3
https://huggingface.co/facebook/sam3
```

Local repo path:

```text
D:\projects\sam3
```

Local weight:

```text
D:\projects\sam3\checkpoint\sam3.pt
```

CO3Dv2 main experiments must use CO3Dv2 file masks by default:

```yaml
frontend:
  mask_source: file
```

SAM3 is only used for optional automatic-mask diagnostics, such as comparing SAM3 masks against CO3Dv2 masks.

Do not use `sam3.1_multiplex.pt` as the default for CO3Dv2 single-image diagnostics.

---

## 5. Current Project Status

The codebase already includes:

```text
minimal pipeline
real input support
same-mask protocol
file mask loader
dataset runner
ablation switches
baseline protocol
dummy baselines
independent Gaussian baseline
comparison runner
reporting and diagnostics
external baseline adapter
run registry
stress benchmark
object editing suite
experiment orchestration
paper experiment pack
artifact validation
CO3Dv2 single converter
DINOv3 dense feature extraction fix
Windows RTX 5090 GPU runtime
```

Confirmed behavior from recent logs:

```text
CO3Dv2 benchmark conversion works
CO3Dv2 benchmark validation works
DINOv3 ViT-S/16 dense feature extraction works
frontend cache with DINOv3 ViT-S/16 can validate with num_valid=5, num_failed=0
GPU runtime works
```

Known non-final/debug warnings that must be addressed before paper claims:

```text
ToyShapeBank fallback is not paper-final
SoftGaussianRenderer fallback is not paper-final
debug iteration cap is not paper-final
example_dataset is not paper-final
CO3Dv2 single is diagnostic, not the multi-object main benchmark
```

---

## 6. Paper Method and Innovation Points

The main technical route is:

```text
SAM3-DINOv3 frozen front-end
+ visible-hidden object Gaussian buffers
+ scene-coupled ownership optimization
+ confidence-weighted hidden support prior
+ differentiable edit-consistency optimization
```

The role of SAM3-DINOv3 is limited to:

```text
retained visible mask construction
dense feature extraction
mask-guided instance descriptor construction
shape retrieval descriptor construction
```

SAM3-DINOv3 is not the main algorithmic contribution.

### 6.1 Input Processing

For each input image:

1. Obtain retained visible masks.
   - Main same-mask benchmark: file masks.
   - Optional automatic diagnostic: SAM3.
2. Extract DINOv3 dense features.
3. Pool dense DINOv3 features inside each retained mask to obtain instance descriptors.
4. Optionally estimate weak monocular depth.
5. Initialize visible Gaussian buffers from visible masks and depth.
6. Retrieve shape prior candidates using instance descriptors.
7. Initialize hidden Gaussian buffers from retrieved priors.
8. Optimize scene-coupled object ownership and reconstruction losses.
9. Run differentiable edit-consistency optimization.
10. Save Ours output compatible with baseline protocol.

### 6.2 Core Innovations

The four core innovations are:

1. **Visible-hidden factorized Gaussian object representation**

   Each foreground object is represented using separated visible and hidden Gaussian buffers. Visible buffers explain observed pixels; hidden buffers support plausible completion and editing.

2. **Scene-coupled object ownership rendering**

   Objects are not reconstructed independently. Rendering returns per-object contribution and ownership maps, enabling scene-level competition and reducing identity leakage, background leakage, and object swaps.

3. **Confidence-weighted hidden support prior**

   DINOv3 mask-pooled descriptors retrieve shape support candidates. Retrieval confidence controls hidden Gaussian budget and hidden prior strength. The prior is soft, not a hard template.

4. **Differentiable edit-consistency optimization**

   Object-level edits such as remove, isolate, translate, and scale are used to regularize object decomposition. Non-edited regions should remain stable, while edited object regions should change consistently.

---

## 7. Debug vs Paper-Ready Results

### 7.1 Debug-only results

The following are not paper-final:

```text
examples/example_dataset
configs/minimal.yaml results
DinoV3Stub
Sam3Stub
DepthStub
ToyShapeBank
SoftGaussianRenderer fallback
3/3/3/2 debug iteration cap
dummy baselines only
CO3Dv2 single as multi-object benchmark
```

These can be used for:

```text
smoke tests
pipeline verification
debug visualization
engineering diagnostics
```

### 7.2 Paper-ready requirements

For NeurIPS / CVPR-level experimental tables, use:

```text
fixed benchmark manifest
same-mask protocol
official DINOv3 weights
file masks or cached SAM3 masks fixed before comparison
real or prepared shape bank
no ToyShapeBank fallback for main claims
renderer setting explicitly reported
real baselines or validated baseline outputs
no debug iteration cap
clear geometry metric availability
```

If geometry GT is absent, do not report Chamfer / F-score.

---

## 8. Current Immediate Direction

The project should not continue adding new minimal/debug framework modules.

The immediate code task is:

```text
v3.6.2-co3dv2-highres-debug-fix
```

It must fix low-resolution CO3Dv2 outputs by:

```text
adding high-res configs
fixing image resize
fixing mask resize
rebuilding high-res frontend cache
saving resolution diagnostics
disabling or making explicit debug iteration cap
```

---

## 9. Command Style

Good:

```bat
python scripts/check_dinov3_dependencies.py
```

Good:

```bat
python scripts/cache_co3dv2_real_frontend.py --config configs/co3dv2_real_frontend_debug.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16 --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16/cache_manifest.csv --validate --require-cuda --check-deps-first
```

Avoid PowerShell-only style:

```powershell
python scripts/foo.py `
  --arg value
```

Avoid Bash heredoc:

```bash
python - <<'PY'
```

---

## 10. Common Commands

### 10.1 Activate environment

```bat
conda activate shapesplat
cd /d C:\Users\zhaoc\ShapeSplat
```

### 10.2 Project health

```bat
python scripts/check_project_health.py
python scripts/run_quick_tests.py
```

### 10.3 GPU runtime

```bat
python scripts/print_gpu_info.py
python scripts/check_gpu_runtime.py --config configs/local_windows_rtx5090.yaml --device cuda --require-cuda --out outputs/check_gpu_runtime_cuda
python scripts/run_gpu_smoke_experiment.py --config configs/local_windows_rtx5090.yaml --out outputs/gpu_smoke_cuda --require-cuda --iters 2
```

### 10.4 CO3Dv2 inspect / convert / validate

```bat
python scripts/inspect_co3dv2_single.py --root D:\projects\datasets\co3dv2_single --out outputs/inspect_co3dv2_single
python scripts/convert_co3dv2_single.py --root D:\projects\datasets\co3dv2_single --out data/co3dv2_single_benchmark --max-categories 3 --max-sequences 2 --max-frames-per-sequence 5 --copy-files --overwrite
python scripts/validate_benchmark_v2.py --manifest data/co3dv2_single_benchmark/manifest.csv --config configs/final_benchmark.yaml --out outputs/validate_co3dv2_single
```

### 10.5 DINOv3 dependency check

```bat
python scripts/check_dinov3_dependencies.py
```

If required dependencies are missing:

```bat
python -m pip install torchmetrics omegaconf ftfy regex scikit-learn submitit termcolor
```

### 10.6 DINOv3 ViT-S/16 check

```bat
python scripts/check_dinov3_weights.py --config configs/co3dv2_real_frontend_debug.yaml --input examples/test_image.png --out outputs/check_dinov3_vits16 --device cuda
```

### 10.7 CO3Dv2 frontend cache

```bat
python scripts/cache_co3dv2_real_frontend.py --config configs/co3dv2_real_frontend_debug.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16 --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16/cache_manifest.csv --validate --require-cuda --check-deps-first
```

### 10.8 CO3Dv2 Ours benchmark

```bat
python scripts/run_ours_benchmark.py --config configs/final_ours.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_debug --max-images 5 --use-frontend-cache --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16/cache_manifest.csv
```

### 10.9 High-resolution CO3Dv2 expected commands

After implementing v3.6.2:

```bat
python scripts/cache_co3dv2_real_frontend.py --config configs/co3dv2_real_frontend_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out-cache outputs/cache_co3dv2_real_frontend_vits16_highres --max-images 5 --write-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv --validate --require-cuda --check-deps-first
```

```bat
python scripts/run_ours_benchmark.py --config configs/final_ours_co3dv2_highres.yaml --manifest data/co3dv2_single_benchmark/manifest.csv --out outputs/ours_co3dv2_vits16_highres --max-images 5 --use-frontend-cache --frontend-cache-manifest outputs/cache_co3dv2_real_frontend_vits16_highres/cache_manifest.csv
```

---

## 11. Code Modification Rules

### 11.1 Do not import scripts in tests

Tests must not import from `scripts`.

Bad:

```python
from scripts.run_x import foo
```

Good:

```python
from shapesplat.experiments.x import foo
```

Reusable logic must live under:

```text
src/shapesplat/
```

Scripts are CLI wrappers only.

### 11.2 Keep default compatibility

Do not break:

```text
configs/minimal.yaml
scripts/run_minimal.py
scripts/run_experiment.py
tests/test_smoke.py
tests/test_real_input.py
tests/test_evaluation.py
```

### 11.3 No hidden heavyweight dependencies

Do not introduce mandatory dependencies on:

```text
pandas
open3d
trimesh
PyTorch3D
LPIPS
diff_gaussian_rasterization
gsplat
```

unless explicitly requested.

### 11.4 Real backends must be optional

The project must remain importable and testable without:

```text
SAM3 repo
DINOv3 repo
Depth model
real 3DGS renderer
external baselines
```

Real backend failure should produce clear diagnostics, not obscure tracebacks.

### 11.5 Do not silently accept bad paper settings

For final/paper configs, readiness checks should warn or fail for:

```text
stub DINO/SAM/Depth
ToyShapeBank
SoftRenderer fallback
missing external baselines
debug iteration cap
example_dataset used as final benchmark
no geometry GT when geometry metrics are requested
```

---

## 12. Visualization and Resolution Rules

CO3Dv2 raw validation has shown image/mask sizes around:

```text
image_shape: [3, 640, 479]
mask_shape: [1, 640, 479]
```

Do not downsample CO3Dv2 to `128x128` for real diagnostics.

For CO3Dv2 high-resolution diagnostics:

```text
image long side should be around 640
DINO input should be at least 448 for debug
mask resize must use nearest
per-sample directories should save fullres and working-resolution visualizations
report thumbnails must be labeled as thumbnails
```

Required diagnostics fields:

```text
original_image_shape
original_mask_shape
working_image_shape
working_mask_shape
renderer_image_shape
frontend_cache_used
frontend_cache_dir
mask_source
mask_resize_applied
mask_resize_mode
dino_input_size
dino_feature_shape
debug_iteration_cap_applied
visible_steps
hidden_steps
joint_steps
edit_steps
shape_bank_backend
renderer_backend
```

---

## 13. Current Known Issues and Fixes

### 13.1 DINOv3 dependency missing

Symptom:

```text
No module named 'torchmetrics'
```

Fix:

```bat
python -m pip install torchmetrics omegaconf ftfy regex scikit-learn submitit termcolor
```

### 13.2 DINOv3 returns `[1,384]`

Symptom:

```text
Unsupported DINO tensor shape: (1, 384)
```

Cause:

```text
global image embedding was returned instead of dense patch features
```

Fix:

```text
use get_intermediate_layers or forward_features
extract patch tokens
reshape to [D,H,W]
do not use [B,D] as dense feature
```

### 13.3 Low-resolution CO3Dv2 output

Symptom:

```text
input image looks 128x128
mask is polygonal / rough
render is blurry
```

Cause:

```text
minimal 128 resize
low-res frontend cache
mask downsampling
debug iteration cap
SoftRenderer fallback
```

Fix:

```text
use CO3Dv2 high-res configs
rebuild high-res frontend cache
use nearest mask resize
disable debug iteration cap
save diagnostics for original/working/render resolutions
```

### 13.4 Multi-GSO manifest missing

Symptom:

```text
FileNotFoundError: data\multi_gso_same_mask\manifest.csv
```

Cause:

```text
Multi-GSO benchmark was not prepared
```

Fix:

```text
convert or create benchmark manifest before running Multi-GSO commands
```

---

## 14. Versioning Rules

After meaningful changes:

```bat
git add .
git commit -m "<version message>"
git tag <version-tag>
```

Suggested current tags:

```text
v3.6-co3dv2-real-frontend-cache-pack
v3.6.1-fix-dinov3-dense-feature-extraction
v3.6.2-co3dv2-highres-debug-fix
```

Do not tag a version until the requested tests and commands pass.

---

## 15. What to Work on Next

The next immediate task is:

```text
v4.0-full-project-completion-on-existing-code
```

This should complete the CO3Dv2 high-resolution diagnostic workflow on top of the existing full project code by:

```text
using configs/co3dv2_real_frontend_highres.yaml
using configs/final_ours_co3dv2_highres.yaml
checking high-res readiness
rebuilding high-res frontend cache
running Ours and variants with cached DINOv3 descriptors
inspecting output resolution diagnostics
generating CO3Dv2 high-res diagnostic report
```

Do not continue adding new minimal/debug framework modules before this workflow is stable.
