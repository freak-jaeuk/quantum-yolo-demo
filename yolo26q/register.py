"""
Register QuantumC2PSA so it SURVIVES m.train().

Bug recap: replacing model.model[10] with a Python object was undone because
m.train() rebuilds the model from yaml via parse_model(). parse_model resolves
classes from tasks globals() and routes args by `if m in base_modules` /
`if m in repeat_modules` (frozenset identity membership).

Fix strategy (robust): rebuild those two frozensets to INCLUDE QuantumC2PSA by
wrapping parse_model. We can't mutate the local frozensets, so we patch the
module-level names the function closes over... but they're locals. So instead
we wrap parse_model and, before calling the original, temporarily alias the
yaml module string "QuantumC2PSA" handling by monkeypatching the frozensets via
a re-exec is overkill. Simplest reliable approach: subclass C2PSA AND add the
subclass to the frozensets by reconstructing parse_model's globals is not
possible either.

PRACTICAL FIX USED HERE:
  - QuantumC2PSA subclasses C2PSA (same __init__ signature c1,c2,n,e).
  - We monkeypatch parse_model so that the two membership frozensets are
    extended with QuantumC2PSA at call time. We do this by replacing
    tasks.parse_model with a version whose bytecode-level frozensets contain
    our class — achieved by re-defining the relevant frozensets through a
    lightweight wrapper that pre-processes the yaml: any layer whose module is
    "QuantumC2PSA" is built by temporarily swapping to C2PSA for arg parsing,
    then the actual instance is upgraded to QuantumC2PSA afterwards.
"""

import torch
import torch.nn as nn
import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules import C2PSA

from models.quantum_layers import VariationalLayer


class QuantumChannelAttn(nn.Module):
    def __init__(self, channels, n_qubits=4, n_vqc_layers=4):
        super().__init__()
        self.dim = 2 ** n_qubits
        self.down = nn.Linear(channels, self.dim, bias=False)
        self.vqc = VariationalLayer(n_qubits, n_layers=n_vqc_layers)
        self.up = nn.Linear(self.dim, channels, bias=False)
        # Init as IDENTITY: up=0 -> gate factor exp(0)=1 -> qgate(x)=x at start,
        # so the pretrained C2PSA output is preserved and fine-tune only adds
        # the quantum modulation gradually (avoids destroying pretrained scale).
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        g = x.mean(dim=[2, 3])
        d = self.down(g)              # linear (matches x dtype under AMP)
        q = self.vqc(d.float())       # VQC needs float32 (complex internals)
        # multiplicative gate centered at 1.0 (identity at init since up=0)
        gate = torch.exp(self.up(q.to(d.dtype)))
        return x * gate.unsqueeze(-1).unsqueeze(-1)


class QuantumC2PSA(C2PSA):
    """C2PSA + VQC channel gate. Inherits c1,c2,n,e signature. (additive, ~same params)"""
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        self.qgate = QuantumChannelAttn(c2, n_qubits=4, n_vqc_layers=4)

    def forward(self, x):
        return self.qgate(super().forward(x))


class VQCChannelMixer(nn.Module):
    """
    Lightweight replacement for the PSABlock FFN (128->256->128, ~66K params).
    Instead of a wide 2-layer MLP, do: pointwise down -> VQC mix -> pointwise up,
    with a multiplicative VQC gate. Far fewer params (~2*c*2^n + 1x1 dwconv).
    Keeps a residual so it can represent identity early.
    """
    def __init__(self, c, n_qubits=4, n_vqc_layers=4):
        super().__init__()
        self.dim = 2 ** n_qubits
        self.down = nn.Linear(c, self.dim, bias=False)
        self.vqc = VariationalLayer(n_qubits, n_layers=n_vqc_layers)
        self.up = nn.Linear(self.dim, c, bias=False)
        nn.init.zeros_(self.up.weight)   # identity at init (gate=1)
        # cheap local mixing to compensate for removed FFN capacity
        self.pw = nn.Conv2d(c, c, 1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        g = x.mean(dim=[2, 3])
        gate = torch.exp(self.up(self.vqc(self.down(g).float()).to(x.dtype)))
        y = x * gate.unsqueeze(-1).unsqueeze(-1)
        return x + self.bn(self.pw(y))


class QuantumC2PSALight(C2PSA):
    """
    Lightweight quantum C2PSA: keeps cv1/cv2 + attention, but REPLACES each
    PSABlock's heavy FFN with a VQCChannelMixer. This actually REDUCES params
    (the original goal: compression), while the VQC provides channel mixing.
    """
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        # self.c is hidden dim; each block in self.m has .ffn -> replace it
        for blk in self.m:
            if hasattr(blk, "ffn"):
                blk.ffn = VQCChannelMixer(self.c, n_qubits=4, n_vqc_layers=4)

    # forward inherited from C2PSA (uses self.m blocks, now with VQC ffn)


_REGISTERED = False


def register_quantum_modules():
    """
    Patch parse_model so 'QuantumC2PSA' in a yaml is handled exactly like C2PSA
    for arg routing, then instantiated as QuantumC2PSA.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    # Spatial-preserving bottleneck variants for the Pareto-gate experiment.
    from yolo26q.bottleneck_vqc import (
        QBottleneckC2PSA, CBottleneckC2PSA, LBottleneckC2PSA, IBottleneckC2PSA,
        OBottleneckC2PSA, UBottleneckC2PSA, RFFBottleneckC2PSA,
    )
    Q_CLASSES = {
        "QuantumC2PSA": QuantumC2PSA,
        "QuantumC2PSALight": QuantumC2PSALight,
        "QBottleneckC2PSA": QBottleneckC2PSA,
        "CBottleneckC2PSA": CBottleneckC2PSA,
        "LBottleneckC2PSA": LBottleneckC2PSA,
        "IBottleneckC2PSA": IBottleneckC2PSA,
        "OBottleneckC2PSA": OBottleneckC2PSA,      # real-orthogonal control
        "UBottleneckC2PSA": UBottleneckC2PSA,      # complex-unitary control
        "RFFBottleneckC2PSA": RFFBottleneckC2PSA,  # RFF / spectral control
    }
    for name, cls in Q_CLASSES.items():
        setattr(tasks, name, cls)

    import functools
    orig = tasks.parse_model

    @functools.wraps(orig)
    def parse_model_q(d, ch, verbose=True):
        # Rewrite yaml: temporarily present any quantum C2PSA variant AS C2PSA so
        # the original parse_model routes c1,c2,n correctly; record (idx, class).
        import copy
        d2 = copy.deepcopy(d)
        q_positions = []  # (idx, target_class)
        combined = d2["backbone"] + d2["head"]
        for idx, layer in enumerate(combined):
            mod = layer[2]
            if mod in Q_CLASSES:
                q_positions.append((idx, Q_CLASSES[mod]))
                layer[2] = "C2PSA"  # route args as C2PSA
        nb = len(d2["backbone"])
        d2["backbone"] = combined[:nb]
        d2["head"] = combined[nb:]

        model, save = orig(d2, ch, verbose=verbose)

        # Upgrade the built C2PSA at q_positions to the quantum variant in-place.
        for idx, cls in q_positions:
            old = model[idx]
            c2 = old.cv2.conv.out_channels
            c1 = old.cv1.conv.in_channels
            new = cls(c1, c2)
            for attr in ("i", "f", "type", "np"):
                if hasattr(old, attr):
                    setattr(new, attr, getattr(old, attr))
            new.type = cls.__name__
            model[idx] = new
        return model, save

    parse_model_q._quantum_patched = True
    tasks.parse_model = parse_model_q
    _REGISTERED = True
