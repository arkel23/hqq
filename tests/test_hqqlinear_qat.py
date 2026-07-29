# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for HQQLinear's new opt-in activation-quantization and trainable
(QAT) modes - see hqq/core/quantize.py's HQQLinear.__init__ docstring.

Deliberately CPU + plain nn.Linear only: no model download, no GPU, runs in seconds, so
the numerical contract can be checked in isolation before anything touches a real ASR
model. Run:

    python hqq/tests/test_hqqlinear_qat.py

Install the package first so `import hqq` resolves to this checkout rather than any
pip-installed copy:

    pip install -e .

No pytest dependency - plain asserts, exits non-zero on the first failure.

Note: the cross-check that this module's `fake_quant_activation` matches HQQLinearV2's copy
lives in test_transitional_hqqv2_act_parity.py, deliberately NOT here - it exists only for
the duration of the migration and should be deleted along with hqq/core/quantize_v2.py.
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

import torch
from torch import nn

from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

import hqq as _hqq
# hqq may also be pip-installed. Without an editable install of THIS repo, `import hqq`
# resolves to site-packages and the suite would silently test the wrong code.
assert Path(_hqq.__file__).resolve().is_relative_to(_REPO_ROOT), (
    "importing the wrong hqq: %s (expected under %s) - run from the repo root"
    % (_hqq.__file__, _REPO_ROOT)
)

DEV = "cpu"
DTYPE = torch.float32
IN, OUT, GROUP, NBITS = 128, 64, 32, 4


def make_linear(seed=0):
    torch.manual_seed(seed)
    return nn.Linear(IN, OUT, bias=True).to(device=DEV, dtype=DTYPE)


def cfg(nbits=NBITS, group_size=GROUP, optimize=True):
    # BaseQuantizeConfig hardcodes optimize=True inside weight_quant_params and does not
    # expose it as a kwarg, so set it after the fact.
    c = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1)
    c["weight_quant_params"]["optimize"] = optimize
    return c


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


# ---------------------------------------------------------------------------
# 1. Back-compat: defaults off must reproduce the original layer exactly
# ---------------------------------------------------------------------------
def test_defaults_unchanged():
    lin = make_linear()
    a = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    b = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    x = torch.randn(4, IN, dtype=DTYPE)
    a.eval(); b.eval()
    ya, yb = a(x), b(x)
    assert torch.equal(ya, yb), "two identically-built HQQLinears disagree"
    assert a.trainable is False and a.act_bits is None
    assert a.W_q is not None and a.meta is not None, "default path must still pack weights"
    # and the dequantized weight must still track the original
    err = rel_err(a.dequantize().float(), lin.weight.float())
    assert err < 0.2, f"default 4-bit reconstruction unexpectedly poor: {err}"
    print(f"  defaults unchanged: packed, deterministic, recon rel_err={err:.4f}")


# ---------------------------------------------------------------------------
# 2. The point of the change: a trainable layer is calibrated with real HQQ
# ---------------------------------------------------------------------------
def test_trainable_uses_real_hqq_calibration():
    src = make_linear()
    opt_on = HQQLinear(make_linear(), cfg(optimize=True), compute_dtype=DTYPE, device=DEV,
                       del_orig=False, trainable=True)
    opt_off = HQQLinear(make_linear(), cfg(optimize=False), compute_dtype=DTYPE, device=DEV,
                        del_orig=False, trainable=True)
    assert hasattr(opt_on, "calib_scale") and hasattr(opt_on, "calib_zero")
    # optimize=True must actually change the cached scale/zero vs plain min/max, otherwise
    # the half-quadratic solver is not running at all (the original v2 complaint).
    same_scale = torch.allclose(opt_on.calib_scale, opt_off.calib_scale)
    same_zero = torch.allclose(opt_on.calib_zero, opt_off.calib_zero)
    assert not (same_scale and same_zero), (
        "optimize=True produced identical scale/zero to optimize=False - HQQ's proximal "
        "solver is not being applied at calibration time"
    )
    # and it should reconstruct the true weight better
    opt_on.eval(); opt_off.eval()
    w_true = src.weight.float()
    e_on = rel_err(opt_on._fake_quant_weight(opt_on.master_weight, opt_on.calib_scale, opt_on.calib_zero).float(), w_true)
    e_off = rel_err(opt_off._fake_quant_weight(opt_off.master_weight, opt_off.calib_scale, opt_off.calib_zero).float(), w_true)
    assert e_on < e_off, f"optimize=True ({e_on:.5f}) not better than optimize=False ({e_off:.5f})"
    print(f"  HQQ calibration active: rel_err optimize=True {e_on:.5f} < False {e_off:.5f}")


# ---------------------------------------------------------------------------
# 3. eval() reuses the cached calibration; train() recomputes per step
# ---------------------------------------------------------------------------
def test_eval_frozen_train_dynamic():
    layer = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV,
                      del_orig=False, trainable=True)
    x = torch.randn(4, IN, dtype=DTYPE)

    layer.eval()
    y1 = layer(x)
    y2 = layer(x)
    assert torch.equal(y1, y2), "eval() forward is not deterministic"

    # Perturb the weights WITHOUT recalibrating. eval() must keep using the cached
    # scale/zero (that is the whole point), so the cached buffers must not move.
    scale_before = layer.calib_scale.clone()
    with torch.no_grad():
        layer.master_weight.add_(torch.randn_like(layer.master_weight) * 0.05)
    _ = layer(x)
    assert torch.equal(scale_before, layer.calib_scale), "eval() forward mutated the calibration"

    # recalibrate() must move them
    layer.recalibrate()
    assert not torch.equal(scale_before, layer.calib_scale), "recalibrate() did not update scale"
    print("  eval() frozen + recalibrate() refreshes calibration")


# ---------------------------------------------------------------------------
# 4. Gradients actually reach the fp master weight (v1 could never do this)
# ---------------------------------------------------------------------------
def test_gradients_reach_master_weight():
    layer = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV,
                      del_orig=False, trainable=True)
    layer.train()
    x = torch.randn(8, IN, dtype=DTYPE)
    target = torch.randn(8, OUT, dtype=DTYPE)

    loss0 = ((layer(x) - target) ** 2).mean()
    loss0.backward()
    assert layer.master_weight.grad is not None, "no grad on the master weight"
    gnorm = layer.master_weight.grad.norm().item()
    assert gnorm > 0, "master weight grad is all zeros"
    assert torch.isfinite(layer.master_weight.grad).all(), "non-finite grads"

    # a few real optimizer steps must reduce the loss
    opt = torch.optim.SGD(layer.parameters(), lr=1e-2)
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print(f"  grads flow (|g|={gnorm:.4f}); loss {losses[0]:.4f} -> {losses[-1]:.4f} over 60 steps")


# ---------------------------------------------------------------------------
# 5. A non-trainable HQQLinear still cannot train its own weight (unchanged)
# ---------------------------------------------------------------------------
def test_nontrainable_has_no_weight_grad():
    layer = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    assert not hasattr(layer, "master_weight") or not isinstance(getattr(layer, "weight", None), nn.Parameter)
    x = torch.randn(4, IN, dtype=DTYPE, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    # gradient still flows THROUGH to the input, which is what v1 was always able to do
    assert x.grad is not None and torch.isfinite(x.grad).all(), "no gradient through to input"
    print("  non-trainable layer: passes grad through to input, owns no weight grad")


# ---------------------------------------------------------------------------
# 6. freeze() produces a normal packed layer matching a fresh HQQLinear
# ---------------------------------------------------------------------------
def test_freeze_matches_fresh_hqqlinear():
    lin = make_linear(seed=3)
    layer = HQQLinear(make_linear(seed=3), cfg(), compute_dtype=DTYPE, device=DEV,
                      del_orig=False, trainable=True)
    layer.freeze()
    assert layer.trainable is False
    assert layer.W_q is not None and layer.meta is not None, "freeze() did not pack"
    assert not hasattr(layer, "master_weight"), "freeze() left the fp master weight behind"

    fresh = HQQLinear(make_linear(seed=3), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    w_frozen, w_fresh = layer.dequantize().float(), fresh.dequantize().float()
    err = rel_err(w_frozen, w_fresh)
    assert err < 1e-6, f"frozen layer differs from a fresh HQQLinear: rel_err={err}"

    layer.eval(); fresh.eval()
    x = torch.randn(4, IN, dtype=DTYPE)
    assert rel_err(layer(x), fresh(x)) < 1e-6, "frozen forward differs from fresh"
    print(f"  freeze() == fresh HQQLinear (weight rel_err={err:.2e})")


# ---------------------------------------------------------------------------
# 7. Activation quantization: opt-in, dynamic, and independent of trainable
# ---------------------------------------------------------------------------
def test_activation_quantization():
    x = torch.randn(4, IN, dtype=DTYPE)
    plain = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    a8 = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False,
                   act_bits=8)
    plain.eval(); a8.eval()
    y_plain, y_a8 = plain(x), a8(x)
    d = rel_err(y_a8, y_plain)
    assert d > 0, "act_bits=8 changed nothing - activation quantization is not applied"
    assert d < 0.05, f"8-bit activations perturbed the output implausibly much: {d}"

    # lower bit-width must perturb more
    a2 = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False,
                   act_bits=2)
    a2.eval()
    d2 = rel_err(a2(x), y_plain)
    assert d2 > d, f"2-bit acts ({d2:.4f}) not worse than 8-bit ({d:.4f})"

    # grouped activations are accepted and differ from per-tensor
    ag = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False,
                   act_bits=8, act_group_size=32)
    ag.eval()
    assert not torch.equal(ag(x), y_a8), "act_group_size had no effect"

    # works together with trainable
    both = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False,
                     act_bits=8, trainable=True)
    both.train()
    loss = both(x).sum()
    loss.backward()
    assert both.master_weight.grad is not None and torch.isfinite(both.master_weight.grad).all()
    print(f"  act quant: 8-bit rel_diff={d:.5f} < 2-bit {d2:.5f}; grouped differs; QAT+act ok")


# ---------------------------------------------------------------------------
# 8. Settings can arrive via the quant_config dict - which is what makes an
#    UNMODIFIED transformers.HqqConfig able to carry them (see __init__'s comment)
# ---------------------------------------------------------------------------
def test_settings_via_quant_config_dict():
    # This is exactly the shape HqqConfig produces and hands to HQQLinear.
    c = cfg()
    c["act_bits"] = 8
    c["trainable"] = True
    layer = HQQLinear(make_linear(), c, compute_dtype=DTYPE, device=DEV, del_orig=False)
    assert layer.trainable is True, "trainable not picked up from quant_config"
    assert layer.act_bits == 8, "act_bits not picked up from quant_config"
    assert hasattr(layer, "calib_scale"), "trainable path did not initialize"
    # the keys must be CONSUMED, or initialize()'s quantize(**quant_config) would have
    # raised on an unexpected kwarg
    assert "act_bits" not in layer.quant_config and "trainable" not in layer.quant_config

    # and it actually trains
    layer.train()
    x = torch.randn(4, IN, dtype=DTYPE)
    ((layer(x)) ** 2).mean().backward()
    assert layer.master_weight.grad is not None and torch.isfinite(layer.master_weight.grad).all()

    # explicit kwargs still override the dict
    c2 = cfg()
    c2["trainable"] = False
    override = HQQLinear(make_linear(), c2, compute_dtype=DTYPE, device=DEV,
                         del_orig=False, trainable=True)
    assert override.trainable is True, "explicit kwarg did not override quant_config"

    # absent keys keep the old defaults
    plain = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    assert plain.trainable is False and plain.act_bits is None
    print("  quant_config dict route works (HqqConfig needs no modification); kwargs override")


# ---------------------------------------------------------------------------
# 9. Survives transformers' HQQLinear.weight monkey-patch (real bug, caught late)
# ---------------------------------------------------------------------------
def test_survives_transformers_weight_property_patch():
    """transformers/quantizers/quantizer_hqq.py does, at import time:

        @property
        def weight(self): return torch.empty(0, dtype=self.compute_dtype, device=self.device)
        HQQLinear.weight = weight

    a compatibility hack for models that read `layer.weight.dtype`. A property is a DATA
    DESCRIPTOR, so it beats instance attribute lookup. Before the fp master was renamed to
    `master_weight`, this made trainable construction die with
    KeyError("attribute 'weight' already exists") inside register_parameter - and had it not
    raised, `layer.weight` would have silently returned that EMPTY tensor instead of the real
    master weight. The first version of these tests never imported that module, so it passed
    while a real HqqConfig load failed. This test applies the patch explicitly so the blind
    spot cannot come back.
    """
    patched = "weight" in HQQLinear.__dict__
    if not patched:
        prop = property(lambda self: torch.empty(0, dtype=self.compute_dtype, device=self.device))
        HQQLinear.weight = prop
    try:
        assert isinstance(HQQLinear.__dict__.get("weight"), property), "patch not in effect"
        layer = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV,
                          del_orig=False, trainable=True)
        # the real master weight must be reachable and NOT be the dummy empty tensor
        assert isinstance(layer.master_weight, nn.Parameter)
        assert layer.master_weight.numel() == IN * OUT, "master weight is not the real tensor"
        assert layer.weight.numel() == 0, "expected the transformers dummy property here"
        layer.train()
        x = torch.randn(4, IN, dtype=DTYPE)
        (layer(x) ** 2).mean().backward()
        assert layer.master_weight.grad is not None and torch.isfinite(layer.master_weight.grad).all()
        layer.freeze()
        assert layer.W_q is not None and not hasattr(layer, "master_weight")
        print("  works with transformers' weight property patched in; master_weight unshadowed")
    finally:
        if not patched:
            del HQQLinear.weight


# ---------------------------------------------------------------------------
# 10. Activation quantization degrades monotonically as bits fall
# ---------------------------------------------------------------------------
def test_act_bits_monotonic():
    """Accuracy must improve with more activation bits - but only within one scheme.

    fake_quant_activation uses two different schemes: 1 and 1.58 bits scale by the mean
    absolute value (BitNet style), while >=2 bits use a symmetric affine scale of
    Qp/max(|x|) with Qp = 2**(nbits-1) - 1. At nbits=2 that gives Qp=1, i.e. levels
    {-2,-1,0,1} scaled by 1/max|x| - one positive level and max-based scaling - which is
    measurably WORSE than ternary. Measured on a fixed normal input:

        1: 0.610   1.58: 0.520   2: 0.787   3: 0.290   4: 0.124   8: 0.007

    So 2-bit is the odd one out. This test asserts monotonicity over 3..8, where the
    scheme is consistent, and pins the 2-bit anomaly explicitly so that a future fix
    shows up as a deliberate test change rather than passing unnoticed.
    """
    x = torch.randn(16, IN, dtype=DTYPE)
    ref = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    ref.eval()
    y_ref = ref(x)

    def err_at(b):
        layer = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV,
                          del_orig=False, act_bits=b)
        layer.eval()
        return rel_err(layer(x), y_ref)

    bits = [3, 4, 5, 6, 7, 8]
    errs = [err_at(b) for b in bits]
    for (b0, e0), (b1, e1) in zip(zip(bits, errs), zip(bits[1:], errs[1:])):
        assert e1 <= e0 + 1e-6, (
            "act_bits=%s (err %.5f) is worse than act_bits=%s (err %.5f) - not monotonic"
            % (b1, e1, b0, e0)
        )
    assert errs[-1] < errs[0] / 10, "8-bit should be far better than 3-bit"

    # the known 2-bit anomaly: worse than ternary, because Qp=1 there
    e2, e158 = err_at(2), err_at(1.58)
    assert e2 > e158, (
        "2-bit is no longer worse than 1.58-bit (%.5f vs %.5f) - the Qp=1 anomaly may have "
        "been fixed; update this test if so" % (e2, e158)
    )
    print("  monotonic 3..8: " + " > ".join("%s:%.4f" % (b, e) for b, e in zip(bits, errs)))
    print("  pinned anomaly: 2-bit %.4f > 1.58-bit %.4f (Qp=1, max-scaled)" % (e2, e158))


# ---------------------------------------------------------------------------
# 11. Weight bit-width degrades monotonically too
# ---------------------------------------------------------------------------
def test_weight_nbits_monotonic():
    src = make_linear()
    w_true = src.weight.float()
    bits = [2, 3, 4, 8]
    errs = []
    for nb in bits:
        layer = HQQLinear(make_linear(), cfg(nbits=nb), compute_dtype=DTYPE, device=DEV,
                          del_orig=False)
        errs.append(rel_err(layer.dequantize().float(), w_true))
    for (b0, e0), (b1, e1) in zip(zip(bits, errs), zip(bits[1:], errs[1:])):
        assert e1 <= e0 + 1e-6, (
            "nbits=%d (err %.5f) worse than nbits=%d (err %.5f)" % (b1, e1, b0, e0)
        )
    print("  monotonic: " + " > ".join("w%d:%.4f" % (b, e) for b, e in zip(bits, errs)))


# ---------------------------------------------------------------------------
# 12. Freeze into canonical form AFTER training reflects the trained weights
# ---------------------------------------------------------------------------
def test_freeze_after_training():
    """The point of the trainable mode: train, then collapse back to a plain packed
    HQQLinear whose weights are the TRAINED ones, not the originals."""
    layer = HQQLinear(make_linear(), cfg(), compute_dtype=DTYPE, device=DEV,
                      del_orig=False, trainable=True)
    w_before = layer.master_weight.detach().clone()

    layer.train()
    x = torch.randn(8, IN, dtype=DTYPE)
    target = torch.randn(8, OUT, dtype=DTYPE)
    opt = torch.optim.SGD(layer.parameters(), lr=1e-2)
    for _ in range(40):
        opt.zero_grad()
        ((layer(x) - target) ** 2).mean().backward()
        opt.step()

    w_trained = layer.master_weight.detach().clone()
    moved = rel_err(w_trained, w_before)
    assert moved > 1e-3, "training did not move the weights (%.2e)" % moved

    layer.freeze()
    assert layer.W_q is not None and not hasattr(layer, "master_weight")

    # a fresh HQQLinear built from the TRAINED weight must match the frozen layer
    ref_lin = nn.Linear(IN, OUT, bias=True).to(device=DEV, dtype=DTYPE)
    with torch.no_grad():
        ref_lin.weight.copy_(w_trained)
    fresh = HQQLinear(ref_lin, cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
    err = rel_err(layer.dequantize().float(), fresh.dequantize().float())
    assert err < 1e-6, "frozen layer does not match a fresh one built from trained weights: %.2e" % err

    # and it must NOT match one built from the ORIGINAL weight
    err_orig = rel_err(layer.dequantize().float(), w_before.float())
    assert err_orig > 1e-3, "frozen layer still reflects the pre-training weight"
    print("  trained (moved %.4f), froze, matches fresh-from-trained (%.1e)" % (moved, err))


def main():
    tests = [
        ("back-compat: defaults unchanged", test_defaults_unchanged),
        ("trainable uses real HQQ calibration", test_trainable_uses_real_hqq_calibration),
        ("eval frozen / recalibrate refreshes", test_eval_frozen_train_dynamic),
        ("gradients reach fp master weight", test_gradients_reach_master_weight),
        ("non-trainable owns no weight grad", test_nontrainable_has_no_weight_grad),
        ("freeze() == fresh HQQLinear", test_freeze_matches_fresh_hqqlinear),
        ("activation quantization", test_activation_quantization),
        ("settings via quant_config dict", test_settings_via_quant_config_dict),
        ("survives transformers weight patch", test_survives_transformers_weight_property_patch),
        ("act_bits monotonic", test_act_bits_monotonic),
        ("weight nbits monotonic", test_weight_nbits_monotonic),
        ("freeze after training", test_freeze_after_training),
    ]
    for name, fn in tests:
        print(f"\n[{name}]")
        fn()
    print("\nALL TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
