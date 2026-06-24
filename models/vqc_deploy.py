"""
Deploy-time compilation of a trained VariationalLayer (VQC) to a pure-classical,
edge-friendly op.

WHY: at training time the VQC is a genuine parameterized quantum circuit — the
unitary U is rebuilt from rotation angles each forward and gradients flow through
the circuit (RY/RZ/CNOT). After training the angles are FROZEN, so U is a fixed
matrix. Re-simulating the circuit (complex kron products, fp32-only, ComplexHalf
issues) at inference is pure overhead and blocks INT8/edge export.

CompiledVQC freezes U once and runs the SAME function with only real ops:
    y = sqrt((s @ Ur)^2 + (s @ Ui)^2) * |x|      where s = x/|x|, U = Ur + i*Ui
This is numerically identical to VariationalLayer.forward (statevector |U s|),
but is two real matmuls + an L2 — exportable to ONNX/NCNN/OpenVINO and runs at
classical speed on a CPU/NPU. The quantum circuit is still what was *trained*;
deployment just uses its learned transform (as you would after running it on a QPU).
"""
import torch
import torch.nn as nn

from models.quantum_layers import VariationalLayer


class CompiledVQC(nn.Module):
    """Frozen-unitary classical equivalent of a trained VariationalLayer."""

    def __init__(self, U: torch.Tensor):
        super().__init__()
        # store U^T split into real/imag so forward is real-only
        self.register_buffer("Ur", U.real.T.contiguous())   # [d, d]
        self.register_buffer("Ui", U.imag.T.contiguous())
        self.dim = U.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: intentionally NOT @torch.no_grad() — the buffers are frozen
        # anyway, but QAT fine-tuning needs gradients to flow THROUGH this op
        # to the layers before it (no_grad would silently detach the down-proj).
        norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        s = x / norms
        re = s @ self.Ur
        im = s @ self.Ui
        # clamp avoids sqrt(0) grad NaN during fine-tuning
        return torch.sqrt((re * re + im * im).clamp_min(1e-12)) * norms


@torch.no_grad()
def compile_vqc(vqc: VariationalLayer) -> CompiledVQC:
    """Build the frozen unitary from the trained circuit and return CompiledVQC."""
    U = vqc.get_unitary().detach()
    return CompiledVQC(U)


@torch.no_grad()
def replace_vqcs_with_compiled(module: nn.Module) -> int:
    """Recursively swap every VariationalLayer in `module` for its CompiledVQC.
    Returns the number replaced. Call AFTER training, before edge export."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, VariationalLayer):
            setattr(module, name, compile_vqc(child))
            n += 1
        else:
            n += replace_vqcs_with_compiled(child)
    return n


# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    for nq in (3, 4, 5):
        vqc = VariationalLayer(nq, n_layers=4).eval()
        # randomize params (simulate a trained circuit)
        with torch.no_grad():
            vqc.ry_params.normal_(0, 1.0)
            vqc.rz_params.normal_(0, 1.0)
        cvqc = compile_vqc(vqc)
        x = torch.randn(32, 2 ** nq)
        a = vqc(x)
        b = cvqc(x)
        err = (a - b).abs().max().item()
        print(f"n_qubits={nq} dim={2**nq}  max|orig-compiled|={err:.2e}  "
              f"{'OK' if err < 1e-4 else 'MISMATCH'}")

    # speed (CPU)
    import time
    vqc = VariationalLayer(4, n_layers=4).eval()
    cvqc = compile_vqc(vqc)
    x = torch.randn(4096, 16)
    for mod, nm in [(vqc, "VQC(sim)"), (cvqc, "CompiledVQC")]:
        t = time.time()
        for _ in range(200):
            mod(x)
        print(f"{nm:<12} 200 iters: {time.time()-t:.3f}s")


if __name__ == "__main__":
    _selftest()
