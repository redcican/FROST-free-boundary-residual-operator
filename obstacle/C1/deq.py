"""
FROST C1 — DEQ / implicit-differentiation upgrade of the forward operator.

The obstacle problem solution is the projected fixed point  u <- max(chi, mean_nbr(u)),  u=0 on ∂.
So instead of a feed-forward FNO, FROST learns an EQUILIBRIUM operator: a fixed-point cell f_theta(z; chi)
whose equilibrium z* = f_theta(z*; chi) decodes to (u, phi). The fixed point is found with Anderson
acceleration; gradients use the IMPLICIT FUNCTION THEOREM (adjoint fixed-point solve), so training is
O(1) in memory w.r.t. the number of solver iterations (no unrolling) -- the DEQ advantage.

This mirrors the physics (the obstacle problem is itself a fixed point) and gives a convergence
diagnostic (the forward residual) the feed-forward FNO cannot. Pure PyTorch / CPU; reuses SpectralConv2d.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from fno import SpectralConv2d


def anderson(f, x0, m=4, lam=1e-4, max_iter=30, tol=1e-3, beta=1.0):
    """Anderson-accelerated fixed-point solve for f(x)=x; returns (x*, residual_history)."""
    B = x0.shape[0]; shape = x0.shape; d = x0[0].numel()
    X = torch.zeros(B, m, d, dtype=x0.dtype, device=x0.device)
    Fv = torch.zeros(B, m, d, dtype=x0.dtype, device=x0.device)
    X[:, 0] = x0.reshape(B, -1); Fv[:, 0] = f(x0).reshape(B, -1)
    X[:, 1] = Fv[:, 0]; Fv[:, 1] = f(Fv[:, 0].view(shape)).reshape(B, -1)
    H = torch.zeros(B, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(B, m + 1, 1, dtype=x0.dtype, device=x0.device); y[:, 0] = 1
    res = []; k = 1
    for k in range(2, max_iter):
        n = min(k, m)
        G = Fv[:, :n] - X[:, :n]
        H[:, 1:n+1, 1:n+1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, device=x0.device)[None]
        alpha = torch.linalg.solve(H[:, :n+1, :n+1], y[:, :n+1])[:, 1:n+1, 0]
        X[:, k % m] = beta * (alpha[:, None] @ Fv[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        Fv[:, k % m] = f(X[:, k % m].view(shape)).reshape(B, -1)
        r = (Fv[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + Fv[:, k % m].norm().item())
        res.append(r)
        if r < tol:
            break
    return X[:, k % m].view(shape), res


class DEQObstacle(nn.Module):
    def __init__(self, modes=12, width=24, f_iter=30, b_iter=20, tol=1e-3):
        super().__init__()
        self.width = width
        self.f_iter, self.b_iter, self.tol = f_iter, b_iter, tol
        self.lift = nn.Linear(1 + 2, width)                # input injection from chi + grid coords
        self.inj = nn.Conv2d(width, width, 1)
        self.cs = SpectralConv2d(width, width, modes, modes)
        self.cw = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)

    @staticmethod
    def _grid(B, H, W, device):
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)

    def cell(self, z, injx):
        return F.gelu(self.cs(z) + self.cw(z) + injx)      # fixed-point map f(z; chi)

    def forward(self, x, return_res=False):
        B, H, W, _ = x.shape
        injx = self.inj(self.lift(torch.cat([x, self._grid(B, H, W, x.device)], -1)).permute(0, 3, 1, 2))
        z0 = torch.zeros_like(injx)
        with torch.no_grad():                              # forward fixed-point solve (no graph)
            zstar, res = anderson(lambda z: self.cell(z, injx), z0,
                                  max_iter=self.f_iter, tol=self.tol)
        z = self.cell(zstar, injx)                         # re-attach one differentiable step
        if torch.is_grad_enabled():                        # implicit (IFT) backward via adjoint solve
            z0d = z.clone().detach().requires_grad_()
            f0 = self.cell(z0d, injx)

            def backward_hook(grad):
                g, _ = anderson(lambda u: torch.autograd.grad(f0, z0d, u, retain_graph=True)[0] + grad,
                                torch.zeros_like(grad), max_iter=self.b_iter, tol=self.tol)
                return g
            z.register_hook(backward_hook)
        out = self.fc2(F.gelu(self.fc1(z.permute(0, 2, 3, 1))))   # (B,H,W,2)
        return (out, res) if return_res else out
