import torch
import numpy as np
import matplotlib.pyplot as plt
import time

def fk_batch(theta1, theta2, l1=1.0, l2=1.0):
    """
    theta1, theta2: (N,) grids of joint angles
    returns: (N, 2) end effector positions
    """
    x2 = l1*torch.cos(theta1) + l2*torch.cos(theta1+theta2)
    y2 = l1*torch.sin(theta1) + l2*torch.sin(theta1+theta2)
    return torch.stack([x2, y2], dim=-1)

def fk_elbow_batch(theta1, l1=1.0):
    return torch.stack([l1*torch.cos(theta1),
                        l1*torch.sin(theta1)], dim=-1)

def build_collision_map(resolution, obstacles, device='cpu'):
    """
    Check every (theta1, theta2) configuration on a grid for collisions.
    resolution: number of grid points per joint angle
    Returns: (resolution, resolution) bool tensor — True = collision
    """
    angles = torch.linspace(-torch.pi, torch.pi, resolution, device=device)
    t1, t2 = torch.meshgrid(angles, angles, indexing='ij')  # (R, R)
    t1_flat = t1.reshape(-1)   # (R*R,)
    t2_flat = t2.reshape(-1)

    elbow = fk_elbow_batch(t1_flat)          # (R*R, 2)
    ee    = fk_batch(t1_flat, t2_flat)        # (R*R, 2)

    collision = torch.zeros(resolution*resolution, dtype=torch.bool, device=device)
    margin = 0.15
    for (cx, cy, r) in obstacles:
        center = torch.tensor([cx, cy], device=device, dtype=torch.float32)
        d_elbow = torch.norm(elbow - center, dim=-1)
        d_ee    = torch.norm(ee    - center, dim=-1)
        collision |= (d_elbow < r + margin)
        collision |= (d_ee    < r + margin)

    return collision.reshape(resolution, resolution)


def benchmark():
    obstacles = [(0.8, 0.4, 0.25), (0.3, 0.9, 0.2), (-0.5, 0.7, 0.2)]

    print(f"{'Resolution':>12} | {'Configs':>10} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>8}")
    print("-" * 62)

    for res in [50, 100, 200, 500, 1000]:
        times = {}
        for device in ['cpu', 'cuda']:
            # Warmup
            build_collision_map(res, obstacles, device=device)
            if device == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            for _ in range(5):
                cm = build_collision_map(res, obstacles, device=device)
            if device == 'cuda':
                torch.cuda.synchronize()
            times[device] = (time.perf_counter() - t0) / 5 * 1000  # ms

        speedup = times['cpu'] / times['cuda']
        print(f"{res:>12} | {res*res:>10,} | {times['cpu']:>10.2f} | "
              f"{times['cuda']:>10.2f} | {speedup:>7.2f}x")

    # Visualize collision map at res=200
    print("\nBuilding collision map at resolution=200...")
    cm_gpu = build_collision_map(200, obstacles, device='cuda').cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: collision map in config space
    axes[0].imshow(cm_gpu, origin='lower', extent=[-180,180,-180,180],
                   cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xlabel('θ₂ (degrees)'); axes[0].set_ylabel('θ₁ (degrees)')
    axes[0].set_title('Collision Map — Configuration Space\n(red=collision, green=free)')

    # Right: obstacles in workspace
    ax = axes[1]
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('Obstacles — Workspace')
    for (cx, cy, r) in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color='red', alpha=0.5))
    ax.scatter(0, 0, color='black', s=100, zorder=5, label='Base')
    ax.legend()

    plt.tight_layout()
    plt.savefig('collision_map.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: collision_map.png")

if __name__ == '__main__':
    print("=" * 62)
    print("GPU-Accelerated Collision Map — 2-Link Arm")
    print("=" * 62)
    benchmark()
