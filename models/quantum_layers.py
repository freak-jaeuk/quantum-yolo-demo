"""
Quantum circuit layers implemented as PyTorch modules.
All quantum operations are simulated via statevector (unitary matrix) operations
on GPU tensors, enabling native PyTorch autograd backpropagation.

Layers:
  1. QFTLayer         — fixed (non-trainable) Quantum Fourier Transform
  2. GroverDiffusionLayer — fixed amplitude amplification (attention-like)
  3. VariationalLayer  — trainable parameterized quantum circuit (SEL)
"""

import math
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# QFT Layer (Fixed — no trainable parameters)
# ---------------------------------------------------------------------------

class QFTLayer(nn.Module):
    """
    Quantum Fourier Transform layer.
    Transforms feature vectors into frequency domain.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.register_buffer("qft_matrix", self._build_qft_matrix(n_qubits))

    @staticmethod
    def _build_qft_matrix(n: int) -> torch.Tensor:
        N = 2 ** n
        phase = 2 * math.pi * torch.arange(N).unsqueeze(1) * torch.arange(N).unsqueeze(0) / N
        F = torch.complex(torch.cos(phase), torch.sin(phase)) / math.sqrt(N)
        return F

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 2^n] → [batch, 2^n]"""
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        state = (x / norms).to(torch.cfloat)
        out = torch.matmul(state, self.qft_matrix.T)
        return out.abs() * norms


# ---------------------------------------------------------------------------
# Grover Diffusion Layer (Fixed — attention-like amplitude amplification)
# ---------------------------------------------------------------------------

class GroverDiffusionLayer(nn.Module):
    """
    Grover diffusion operator: D = 2|s><s| - I
    Amplifies high-amplitude features, suppresses noise.
    """

    def __init__(self, n_qubits: int, n_iterations: int = 1):
        super().__init__()
        dim = 2 ** n_qubits
        D = (2.0 / dim) * torch.ones(dim, dim) - torch.eye(dim)
        self.n_iterations = n_iterations
        self.register_buffer("diffusion_matrix", D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for _ in range(self.n_iterations):
            out = torch.matmul(out, self.diffusion_matrix.T)
        return out


# ---------------------------------------------------------------------------
# Variational Quantum Circuit Layer (Trainable)
# ---------------------------------------------------------------------------

class VariationalLayer(nn.Module):
    """
    Parameterized Quantum Circuit (PQC) with Strongly Entangling Layers.
    Per layer: RY(θ) on each qubit → RZ(φ) on each qubit → CNOT ring
    """

    def __init__(self, n_qubits: int, n_layers: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2 ** n_qubits

        self.ry_params = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)
        self.rz_params = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

        self.register_buffer(
            "cnot_perm", self._build_cnot_ring_permutation(n_qubits)
        )

    @staticmethod
    def _build_cnot_ring_permutation(n_qubits: int) -> torch.Tensor:
        dim = 2 ** n_qubits
        perm = torch.zeros(dim, dim)
        for i in range(dim):
            j = i
            for ctrl in range(n_qubits):
                tgt = (ctrl + 1) % n_qubits
                # Control bit MUST be read from the CURRENT state j (sequential
                # CNOT application = product of true CNOT gates = unitary).
                # Reading from the original i applies all CNOTs "in parallel",
                # which collapses |0..0> and |1..1> together and is NOT unitary.
                if (j >> (n_qubits - 1 - ctrl)) & 1:
                    j = j ^ (1 << (n_qubits - 1 - tgt))
            perm[j, i] = 1.0
        return perm

    def _kronecker_product_ry(self, angles: torch.Tensor) -> torch.Tensor:
        result = torch.eye(1, dtype=torch.cfloat, device=angles.device)
        for i in range(self.n_qubits):
            cos = torch.cos(angles[i] / 2)
            sin = torch.sin(angles[i] / 2)
            ry = torch.stack([
                torch.stack([cos, -sin]),
                torch.stack([sin, cos])
            ]).to(torch.cfloat)
            result = torch.kron(result, ry)
        return result

    def _kronecker_product_rz(self, angles: torch.Tensor) -> torch.Tensor:
        result = torch.eye(1, dtype=torch.cfloat, device=angles.device)
        for i in range(self.n_qubits):
            half = angles[i] / 2
            rz = torch.zeros(2, 2, dtype=torch.cfloat, device=angles.device)
            rz[0, 0] = torch.complex(torch.cos(half), -torch.sin(half))
            rz[1, 1] = torch.complex(torch.cos(half), torch.sin(half))
            result = torch.kron(result, rz)
        return result

    def _build_unitary(self) -> torch.Tensor:
        U = torch.eye(self.dim, dtype=torch.cfloat, device=self.ry_params.device)
        for layer_idx in range(self.n_layers):
            U = self._kronecker_product_ry(self.ry_params[layer_idx]) @ U
            U = self._kronecker_product_rz(self.rz_params[layer_idx]) @ U
            U = self.cnot_perm.to(torch.cfloat) @ U
        return U

    def get_unitary(self) -> torch.Tensor:
        return self._build_unitary()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, 2^n] → [batch, 2^n]"""
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        state = (x / norms).to(torch.cfloat)
        U = self._build_unitary()
        out = torch.matmul(state, U.T)
        return out.abs() * norms
