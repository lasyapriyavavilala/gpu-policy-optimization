import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

def fk(theta, l1=1.0, l2=1.0):
    """theta: (..., 2) -> (..., 2) end effector"""
    x2 = l1*torch.cos(theta[...,0]) + l2*torch.cos(theta[...,0]+theta[...,1])
    y2 = l1*torch.sin(theta[...,0]) + l2*torch.sin(theta[...,0]+theta[...,1])
    return torch.stack([x2, y2], dim=-1)

def fk_elbow(theta, l1=1.0):
    """theta: (..., 2) -> (..., 2) elbow"""
    return torch.stack([l1*torch.cos(theta[...,0]),
                        l1*torch.sin(theta[...,0])], dim=-1)

def batched_optimize(
    B, goal_pos, obstacles,
    T=20, n_iters=500, lr=0.05,
    w_goal=10.0, w_smooth=1.0, w_collision=5.0,
    device='cpu'
):
    """
    Optimize B trajectories simultaneously.
    theta: (B, T, 2) — B independent trajectories in parallel
    """
    # Random initializations for diversity
    theta = torch.randn(B, T, 2, device=device) * 0.5
    theta.requires_grad_(True)

    goal = torch.tensor(goal_pos, device=device, dtype=torch.float32)
    obs  = [(torch.tensor([cx, cy], device=device, dtype=torch.float32), r)
            for cx, cy, r in obstacles]

    optimizer = torch.optim.Adam([theta], lr=lr)

    for _ in range(n_iters):
        optimizer.zero_grad()

        # Goal cost: (B,) — last waypoint reaches goal
        ee_final  = fk(theta[:, -1, :])                    # (B, 2)
        goal_cost = w_goal * ((ee_final - goal)**2).sum(-1) # (B,)

        # Smoothness cost: (B,)
        smooth_cost = w_smooth * ((theta[:,1:,:] - theta[:,:-1,:])**2).sum((-1,-2))

        # Collision cost: (B,)
        col_cost = torch.zeros(B, device=device)
        elbow = fk_elbow(theta.view(B*T, 2)).view(B, T, 2)  # (B, T, 2)
        ee    = fk(theta.view(B*T, 2)).view(B, T, 2)         # (B, T, 2)
        for center, r in obs:
            d_elbow = torch.norm(elbow - center, dim=-1)     # (B, T)
            d_ee    = torch.norm(ee    - center, dim=-1)     # (B, T)
            col_cost += w_collision * (F.relu(r + 0.15 - d_elbow).sum(-1) +
                                       F.relu(r + 0.15 - d_ee).sum(-1))

        loss = (goal_cost + smooth_cost + col_cost).sum()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            theta.clamp_(-torch.pi, torch.pi)

    # Pick best trajectory (lowest goal cost)
    with torch.no_grad():
        ee_final  = fk(theta[:, -1, :])
        goal_cost = ((ee_final - goal)**2).sum(-1)
        best_idx  = goal_cost.argmin()

    return theta.detach(), best_idx.item()


def benchmark():
    goal_pos  = [1.2, 0.8]
    obstacles = [(0.8, 0.4, 0.25), (0.3, 0.9, 0.2)]

    print(f"{'B':>6} | {'CPU (s)':>10} | {'GPU (s)':>10} | {'Speedup':>8}")
    print("-" * 42)

    for B in [1, 8, 32, 128, 512]:
        times = {}
        for device in ['cpu', 'cuda']:
            # Warmup
            batched_optimize(B, goal_pos, obstacles,
                             T=20, n_iters=50, device=device)
            if device == 'cuda':
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            batched_optimize(B, goal_pos, obstacles,
                             T=20, n_iters=300, device=device)
            if device == 'cuda':
                torch.cuda.synchronize()
            times[device] = time.perf_counter() - t0

        speedup = times['cpu'] / times['cuda']
        print(f"{B:>6} | {times['cpu']:>10.3f} | {times['cuda']:>10.3f} | {speedup:>7.2f}x")

    # Final run — best trajectory visualization
    print("\nOptimizing 128 trajectories on GPU for visualization...")
    theta_batch, best_idx = batched_optimize(
        128, goal_pos, obstacles, T=20, n_iters=500, device='cuda'
    )
    best = theta_batch[best_idx].cpu().numpy()

    # Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    ax.set_title('Best trajectory from 128 parallel candidates (GPU)')

    for (cx, cy, r) in obstacles:
        ax.add_patch(plt.Circle((cx, cy), r, color='red', alpha=0.4))
    ax.scatter(*goal_pos, color='green', s=200, zorder=5, marker='*', label='Goal')

    T = len(best)
    for i in range(T):
        t  = best[i]
        x1 = np.cos(t[0]); y1 = np.sin(t[0])
        x2 = x1 + np.cos(t[0]+t[1]); y2 = y1 + np.sin(t[0]+t[1])
        alpha = 0.15 + 0.85 * (i / T)
        lw    = 1.0 if i < T-1 else 3.0
        color = 'blue' if i < T-1 else 'black'
        ax.plot([0, x1, x2], [0, y1, y2], color=color, alpha=alpha, lw=lw)

    ax.scatter(0, 0, color='black', s=100, zorder=5, label='Base')
    ax.legend(); plt.tight_layout()
    plt.savefig('arm_batched_gpu.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: arm_batched_gpu.png")

if __name__ == '__main__':
    print("=" * 50)
    print("Batched Trajectory Optimization — CPU vs GPU")
    print("=" * 50)
    benchmark()
