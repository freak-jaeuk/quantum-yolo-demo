"""
Spatial-preserving VQC bottleneck for the Pareto-gate experiment.

WHY THIS EXISTS (read before changing):
  The earlier QuantumC2PSA put a VQC *channel gate* on the C2PSA output, using
  GAP (x.mean over H,W) to squash spatial info -> it destroyed C2PSA's
  position-sensitive attention and lost -13.6% mAP (and after bug-fixes was a
  wash, noise-level). Lesson: do NOT collapse spatial dims.

  This module instead applies the VQC PER SPATIAL LOCATION on the channel
  vector, so the H x W layout is fully preserved. It is inserted at the deepest
  bottleneck (P5/32, ~20x20) as an additive, identity-at-init residual on top of
  C2PSA. The load-bearing research claim being tested:

    "A VQC mixer in the bottleneck learns a richer compact representation at
     the SAME parameter budget than a classical mixer (Pareto advantage)."

  We test it cleanly: everything (down 1x1, up 1x1, residual, identity-init) is
  IDENTICAL across variants; only the middle transform differs:
    - middle='vqc'     : VariationalLayer unitary  (~ n_layers*n_qubits*2 params)
    - middle='linear'  : full dim x dim linear     (dim^2 params)  [classical, richer]
    - middle='lowrank' : dim -> r -> dim           (2*dim*r params) [classical, param-matched]
    - middle='identity': no middle                 (down/up only; ablates the mixer)

  Pareto read: plot (total_params, mAP). If 'vqc' sits ABOVE the classical
  points despite fewer params -> quantum win. If 'linear' dominates -> classical
  wins (the expected null per our prior findings). 'lowrank' shows what a
  classical mixer does at the VQC's tiny param budget (the fair same-size point).
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules import C2PSA
from models.quantum_layers import VariationalLayer


class OrthogonalMixer(nn.Module):
    """DECISIVE CONTROL for the conditional-advantage study (design Rung 2).

    Same readout as the VQC -- normalize, apply a norm-preserving map, take the
    elementwise magnitude, rescale by the input norm:  |W . x_hat| * ||x||  --
    but W is a CLASSICAL orthogonal/unitary matrix built from a product of
    Householder reflections, NOT a quantum gate ansatz.

      complex=False : real-orthogonal W (W^T W = I). Isolates plain orthogonality
                      + norm-preservation from any quantum/complex structure.
      complex=True  : complex-unitary W (W^H W = I). The TIGHTEST VQC analog --
                      identical sqrt(re^2+im^2) magnitude readout; the ONLY
                      remaining difference vs the VQC is the gate ansatz / the
                      specific reachable subset of unitaries.

    If a VQC effect is matched by complex=True -> it is the unitarity bias, not
    the quantum ansatz. If matched by complex=False -> plain orthogonality.

    Param budget: n_reflections*dim (real) or n_reflections*dim*2 (complex).
    Choose n_reflections to MATCH the VQC's 2*n_layers*n_qubits budget (fair
    same-budget point) or =dim for a full-capacity upper bound. Householder
    products are orthogonal BY CONSTRUCTION; assert_orthogonal() verifies to the
    same 2.4e-7 threshold used to validate the VQC unitary.
    """

    def __init__(self, dim, n_reflections, complex=False):
        super().__init__()
        self.dim = dim
        self.cplx = complex
        self.k = max(1, n_reflections)
        if complex:
            # complex Householder vectors (real+imag parts as params)
            self.vr = nn.Parameter(torch.randn(self.k, dim))
            self.vi = nn.Parameter(torch.randn(self.k, dim))
        else:
            self.v = nn.Parameter(torch.randn(self.k, dim))

    def _matrix(self):
        if self.cplx:
            W = torch.eye(self.dim, dtype=torch.cfloat, device=self.vr.device)
            I = torch.eye(self.dim, dtype=torch.cfloat, device=self.vr.device)
            for i in range(self.k):
                # force fp32->complex64 (never ComplexHalf, even if params are half),
                # mirroring the VQC's explicit .to(cfloat) protection (codex review)
                v = torch.complex(self.vr[i].float(), self.vi[i].float()).unsqueeze(1)
                vhv = (v.conj().T @ v).real.clamp(min=1e-8)
                W = (I - 2.0 * (v @ v.conj().T) / vhv) @ W              # reflection
            return W
        W = torch.eye(self.dim, device=self.v.device)
        I = torch.eye(self.dim, device=self.v.device)
        for i in range(self.k):
            v = self.v[i].unsqueeze(1)                                  # [dim,1]
            vtv = (v.T @ v).clamp(min=1e-8)
            W = (I - 2.0 * (v @ v.T) / vtv) @ W
        return W

    def forward(self, x):
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xn = x / norms
        W = self._matrix()
        if self.cplx:
            out = torch.matmul(xn.to(torch.cfloat), W.T)
        else:
            out = torch.matmul(xn, W.T)
        return out.abs() * norms                # same nonneg magnitude readout as VQC

    @torch.no_grad()
    def assert_orthogonal(self, tol=2.4e-7):
        W = self._matrix()
        I = torch.eye(self.dim, dtype=W.dtype, device=W.device)
        gram = (W.conj().T @ W) if self.cplx else (W.T @ W)
        err = (gram - I).abs().max().item()
        assert err < tol, f"OrthogonalMixer not orthogonal: ||W^HW - I||_max={err:.2e}"
        return err


class RFFMixer(nn.Module):
    """CONTROL for the VQC's truncated-Fourier / band-limited spectral bias
    (design Rung 4). Random Fourier Features are the dequantization literature's
    standard classical surrogate for a trained VQC's Fourier representation.

      x_hat = x/||x|| -> [cos(x_hat.Omega), sin(x_hat.Omega)] -> Linear -> *||x||

    Omega is a FIXED (frozen, non-trainable) random projection, like classic RFF;
    only the readout Linear is trained. 'Frequency-matched' is APPROXIMATE here:
    Omega ~ N(0, sigma^2). A TRUE match to the VQC's frequency comb needs the
    data-re-uploading variant (exploratory; see power-analysis plan). Trainable
    params = 2*n_features*dim. If the VQC effect is matched by RFF -> it is the
    spectral/Fourier bias, classically available.
    """

    def __init__(self, dim, n_features=None, sigma=1.0):
        super().__init__()
        self.dim = dim
        F = dim if n_features is None else n_features
        self.register_buffer("omega", torch.randn(dim, F) * sigma)   # frozen
        self.lin = nn.Linear(2 * F, dim, bias=False)

    def forward(self, x):
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xn = x / norms
        proj = xn @ self.omega
        feat = torch.cat([proj.cos(), proj.sin()], dim=-1)
        return self.lin(feat).abs() * norms       # same readout shape/nonlinearity


class FeatureBottleneck(nn.Module):
    """
    Spatial-preserving bottleneck mixer applied as an identity-init residual.

      x [B,C,H,W] --down 1x1--> [B,dim,H,W]
                  --(per-pixel middle on the dim-vector)-->
                  --up 1x1 (zero-init)--> [B,C,H,W]
      return x + up(middle(down(x)))         # == x at init (up=0)

    The middle runs on every spatial location independently (B*H*W as the batch
    dim), so spatial structure is never pooled away.
    """

    def __init__(self, c, n_qubits=4, n_vqc_layers=4, middle="vqc", lowrank_r=2):
        super().__init__()
        self.middle_kind = middle
        dim = 2 ** n_qubits
        self.dim = dim
        self.down = nn.Conv2d(c, dim, 1, bias=False)
        self.up = nn.Conv2d(dim, c, 1, bias=False)
        nn.init.zeros_(self.up.weight)  # identity at init via the residual

        vqc_params = 2 * n_vqc_layers * n_qubits      # budget to param-match controls to
        if middle == "vqc":
            self.mid = VariationalLayer(n_qubits, n_layers=n_vqc_layers)
        elif middle == "linear":
            self.mid = nn.Linear(dim, dim, bias=False)
        elif middle == "lowrank":
            self.mid = nn.Sequential(
                nn.Linear(dim, lowrank_r, bias=False),
                nn.Linear(lowrank_r, dim, bias=False),
            )
        elif middle == "orthogonal":   # real-orthogonal control, param-matched to VQC
            self.mid = OrthogonalMixer(dim, n_reflections=round(vqc_params / dim),
                                       complex=False)
        elif middle == "unitary":      # complex-unitary control (tightest VQC analog)
            self.mid = OrthogonalMixer(dim, n_reflections=round(vqc_params / (2 * dim)),
                                       complex=True)
        elif middle == "rff":          # spectral/Fourier (RFF) surrogate control
            self.mid = RFFMixer(dim, n_features=dim)
        elif middle == "identity":
            self.mid = None
        else:
            raise ValueError(f"unknown middle: {middle}")
        # complex internals (VQC, unitary) must run fp32 (ComplexHalf NaNs under AMP)
        self.mid_complex = middle in ("vqc", "unitary")

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.down(x)                       # [B, dim, H, W]
        if self.mid is not None:
            # [B,dim,H,W] -> [B*H*W, dim]  (per-pixel channel vectors)
            v = h.permute(0, 2, 3, 1).reshape(-1, self.dim)
            if self.mid_complex:
                # VQC / complex-unitary build a complex matrix; under AMP autocast
                # the complex matmul downcasts to ComplexHalf (experimental,
                # NaN-prone) and collapses training. Force fp32/cfloat here.
                with torch.autocast(device_type=x.device.type, enabled=False):
                    v = self.mid(v.float()).to(h.dtype)
            else:
                v = self.mid(v)
            h = v.reshape(B, H, W, self.dim).permute(0, 3, 1, 2).contiguous()
        return x + self.up(h)


class _BottleneckC2PSA(C2PSA):
    """C2PSA followed by a spatial-preserving FeatureBottleneck.
    Subclasses fix the `middle` kind so register.py can map a yaml string ->
    the right variant. Same (c1,c2,n,e) signature as C2PSA."""
    MIDDLE = "vqc"
    N_QUBITS = 4
    N_VQC_LAYERS = 4
    LOWRANK_R = 2

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.bottleneck = FeatureBottleneck(
            c2, n_qubits=self.N_QUBITS, n_vqc_layers=self.N_VQC_LAYERS,
            middle=self.MIDDLE, lowrank_r=self.LOWRANK_R,
        )

    def forward(self, x):
        return self.bottleneck(super().forward(x))


class QBottleneckC2PSA(_BottleneckC2PSA):
    """VQC bottleneck (the quantum variant under test)."""
    MIDDLE = "vqc"


class CBottleneckC2PSA(_BottleneckC2PSA):
    """Classical full-linear bottleneck (richer middle, more params)."""
    MIDDLE = "linear"


class LBottleneckC2PSA(_BottleneckC2PSA):
    """Classical low-rank bottleneck (param-matched to the VQC, same-size point)."""
    MIDDLE = "lowrank"


class IBottleneckC2PSA(_BottleneckC2PSA):
    """No middle mixer (down/up only) — ablates the mixer, isolates its effect."""
    MIDDLE = "identity"


class OBottleneckC2PSA(_BottleneckC2PSA):
    """DECISIVE control: classical REAL-ORTHOGONAL middle (Householder), same
    readout as VQC, param-matched. If the VQC ties this, the advantage (if any) is
    plain orthogonality+norm-preservation, NOT quantum-specific."""
    MIDDLE = "orthogonal"


class UBottleneckC2PSA(_BottleneckC2PSA):
    """DECISIVE control: classical COMPLEX-UNITARY middle (Householder), the
    tightest VQC analog (identical complex-magnitude readout). If the VQC ties
    this, the only remaining 'quantum' factor is the gate ansatz, not unitarity."""
    MIDDLE = "unitary"


class RFFBottleneckC2PSA(_BottleneckC2PSA):
    """Control for the spectral/Fourier bias: Random-Fourier-Feature middle.
    If the VQC ties this, its useful bias is the (classically reproducible)
    truncated-Fourier representation."""
    MIDDLE = "rff"
