# GPU-Accelerated Policy Inference & Motion Planning

Profiling and optimizing a robot manipulation policy inference loop on NVIDIA T4,
from Python-level analysis down to custom CUDA and Triton kernels — with a
GPU-accelerated motion planner for a 2-link robot arm.

Built as a portfolio project targeting ML systems and inference optimization roles.

---

## Results

| Experiment | Detail | Result |
|---|---|---|
| Roofline analysis | BC policy on RoboMimic Lift task | AI = 0.234 FLOP/byte — memory-bound |
| torch.compile | `reduce-overhead` mode vs eager | **11.8x** latency reduction (2.13ms → 0.18ms) |
| Naive CUDA matmul | Hand-written vs cuBLAS | 317 GFLOP/s — 7.2x behind cuBLAS |
| Tiled CUDA matmul | 16×16 shared memory tiling | 457 GFLOP/s — 1.44x over naive |
| Triton flash-attention | vs PyTorch native SDPA | **1.71x** speedup, numerically exact |
| GPU collision map | 1M configurations, 2-link arm | **23.6x** over CPU (2.79ms vs 65.85ms) |

---

## Project Structure
