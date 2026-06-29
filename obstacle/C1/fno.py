"""
Compact 2D Fourier Neural Operator for FROST C1 (forward operator).

The FROST forward operator maps the obstacle field chi -> (u, phi): it predicts BOTH the membrane
field u AND the level set phi (signed distance to the contact-set free boundary) in one shot. For the
steady obstacle problem this map is the equilibrium of the projected fixed point u <- max(chi,
mean_nbr(u)); here we learn it directly with an FNO (the DEQ/implicit-diff instantiation is the
planned next iteration). Pure PyTorch, CPU-friendly, no external NO library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution: keep the lowest (modes1 x modes2) Fourier modes, learn a complex map."""
    def __init__(self, in_c, out_c, modes1, modes2):
        super().__init__()
        self.in_c, self.out_c, self.m1, self.m2 = in_c, out_c, modes1, modes2
        scale = 1.0 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.rand(in_c, out_c, modes1, modes2, dtype=torch.cfloat))

    @staticmethod
    def _mul(a, b):
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.m1, :self.m2] = self._mul(x_ft[:, :, :self.m1, :self.m2], self.w1)
        out_ft[:, :, -self.m1:, :self.m2] = self._mul(x_ft[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out_ft, s=(H, W))


class FNO2d(nn.Module):
    def __init__(self, modes=16, width=32, in_c=1, out_c=2, n_layers=4):
        super().__init__()
        self.fc0 = nn.Linear(in_c + 2, width)                  # + 2 grid coordinate channels
        self.sp = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_c)

    @staticmethod
    def _grid(shape, device):
        B, H, W = shape
        gx = torch.linspace(0, 1, H, device=device).view(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0, 1, W, device=device).view(1, 1, W, 1).expand(B, H, W, 1)
        return torch.cat([gx, gy], dim=-1)

    def forward(self, x):                                       # x: (B, H, W, in_c)
        x = torch.cat([x, self._grid(x.shape[:3], x.device)], dim=-1)
        x = self.fc0(x).permute(0, 3, 1, 2)
        for sp, w in zip(self.sp, self.w):
            x = F.gelu(sp(x) + w(x))
        x = x.permute(0, 2, 3, 1)
        return self.fc2(F.gelu(self.fc1(x)))                   # (B, H, W, out_c)


class LpLoss:
    """Relative L^p loss, averaged over the batch."""
    def __init__(self, p=2):
        self.p = p

    def __call__(self, x, y):
        B = x.shape[0]
        diff = torch.norm(x.reshape(B, -1) - y.reshape(B, -1), self.p, dim=1)
        ynorm = torch.norm(y.reshape(B, -1), self.p, dim=1)
        return torch.mean(diff / (ynorm + 1e-8))
