# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for HQQLinear's opt-in activation-quantization and trainable (QAT) modes.

Deliberately CPU + plain nn.Linear: no model download, no GPU, runs in seconds. Run from the
repo root, with this checkout on the path so `import hqq` does not resolve to a pip-installed
copy (`pip install -e .`, or PYTHONPATH=$(pwd)):

    python -m unittest tests.test_hqqlinear_qat -v
    python -m unittest tests.test_hqqlinear_qat.TestQAT.test_freeze_matches_fresh -v
"""
import unittest
from pathlib import Path

import torch
from torch import nn

import hqq as _hqq
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig, fake_quant_activation
from hqq.utils.patching import prepare_for_inference

_REPO_ROOT = Path(__file__).resolve().parents[1]
# hqq may also be pip-installed; without this the suite would silently test the wrong code.
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


def cfg(nbits=NBITS, group_size=GROUP, optimize=True, **kwargs):
    # BaseQuantizeConfig hardcodes optimize=True inside weight_quant_params and does not
    # expose it as a kwarg, so set it after the fact.
    c = BaseQuantizeConfig(nbits=nbits, group_size=group_size, axis=1, **kwargs)
    c["weight_quant_params"]["optimize"] = optimize
    return c


def layer(seed=0, **kwargs):
    return HQQLinear(
        make_linear(seed), cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False, **kwargs
    )


def rel_err(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


class SeededTest(unittest.TestCase):
    """Seed per test, so results do not depend on the order the suite happens to run in."""

    def setUp(self):
        torch.manual_seed(1234)


class TestBackCompat(SeededTest):
    """With both flags off, nothing about the layer may have changed."""

    def test_defaults_unchanged(self):
        lin = make_linear()
        a, b = layer(), layer()
        x = torch.randn(4, IN, dtype=DTYPE)
        a.eval(); b.eval()
        self.assertTrue(torch.equal(a(x), b(x)), "two identically-built HQQLinears disagree")
        self.assertIsNone(a.act_bits)
        self.assertFalse(a.trainable)
        self.assertIsNotNone(a.W_q, "default path must still pack weights")
        self.assertIsNotNone(a.meta)
        # a plain layer keeps upstream's zero-indirection cls.forward path
        self.assertNotIn("forward", a.__dict__, "a default layer must not install a forward")
        err = rel_err(a.dequantize().float(), lin.weight.float())
        self.assertLess(err, 0.2, f"default 4-bit reconstruction unexpectedly poor: {err}")
        print(f"\n  defaults: packed, deterministic, no instance forward, recon rel_err={err:.4f}")

    def test_nontrainable_has_no_weight_grad(self):
        """A plain HQQLinear passes gradients through to its input but owns no weight grad -
        that is structural (HQQMatmulNoCacheDeq.backward returns grad_weight=None)."""
        lay = layer()
        self.assertFalse(isinstance(getattr(lay, "weight", None), nn.Parameter))
        x = torch.randn(4, IN, dtype=DTYPE, requires_grad=True)
        lay(x).sum().backward()
        self.assertIsNotNone(x.grad, "no gradient through to input")
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_weight_nbits_monotonic(self):
        w_true = make_linear().weight.float()
        bits = [2, 3, 4, 8]
        errs = [
            rel_err(
                HQQLinear(make_linear(), cfg(nbits=nb), compute_dtype=DTYPE, device=DEV,
                          del_orig=False).dequantize().float(),
                w_true,
            )
            for nb in bits
        ]
        for (b0, e0), (b1, e1) in zip(zip(bits, errs), zip(bits[1:], errs[1:])):
            self.assertLessEqual(e1, e0 + 1e-6, f"nbits={b1} ({e1:.5f}) worse than {b0} ({e0:.5f})")
        print("\n  monotonic: " + " > ".join("w%d:%.4f" % be for be in zip(bits, errs)))


class TestActQuant(SeededTest):
    def test_activation_quantization(self):
        x = torch.randn(4, IN, dtype=DTYPE)
        plain, a8 = layer(), layer(act_bits=8)
        plain.eval(); a8.eval()
        y_plain = plain(x)
        d = rel_err(a8(x), y_plain)
        self.assertGreater(d, 0, "act_bits=8 changed nothing - activation quantization not applied")
        self.assertLess(d, 0.05, f"8-bit activations perturbed the output implausibly much: {d}")

        a2 = layer(act_bits=2); a2.eval()
        d2 = rel_err(a2(x), y_plain)
        self.assertGreater(d2, d, f"2-bit acts ({d2:.4f}) not worse than 8-bit ({d:.4f})")

        grouped = layer(act_bits=8, act_group_size=32); grouped.eval()
        self.assertFalse(torch.equal(grouped(x), a8(x)), "act_group_size had no effect")
        print(f"\n  act quant: 8-bit rel_diff={d:.5f} < 2-bit {d2:.5f}; grouped differs")

    def test_act_quant_works_with_qat(self):
        lay = layer(act_bits=8, trainable=True)
        lay.train()
        lay(torch.randn(4, IN, dtype=DTYPE)).sum().backward()
        self.assertIsNotNone(lay.master_weight.grad)
        self.assertTrue(torch.isfinite(lay.master_weight.grad).all())

    def test_act_bits_monotonic(self):
        """Accuracy improves with more bits, but only within one scheme. 1 and 1.58 bits scale
        by the mean absolute value; >=2 bits use a symmetric affine scale of Qp/max|x|. At 2 bits
        Qp=1, giving levels {-2,-1,0,1} - measurably worse than ternary. Monotonicity is asserted
        over 3..8 where the scheme is consistent; the 2-bit anomaly is pinned separately so a
        future fix shows up as a deliberate test change."""
        x = torch.randn(16, IN, dtype=DTYPE)
        ref = layer(); ref.eval()
        y_ref = ref(x)

        def err_at(b):
            lay = layer(act_bits=b)
            lay.eval()
            return rel_err(lay(x), y_ref)

        bits = [3, 4, 5, 6, 7, 8]
        errs = [err_at(b) for b in bits]
        for (b0, e0), (b1, e1) in zip(zip(bits, errs), zip(bits[1:], errs[1:])):
            self.assertLessEqual(e1, e0 + 1e-6, f"act_bits={b1} ({e1:.5f}) worse than {b0} ({e0:.5f})")
        self.assertLess(errs[-1], errs[0] / 10, "8-bit should be far better than 3-bit")

        e2, e158 = err_at(2), err_at(1.58)
        self.assertGreater(e2, e158, f"2-bit no longer worse than 1.58-bit ({e2:.5f} vs {e158:.5f})")
        print("\n  monotonic 3..8: " + " > ".join("%s:%.4f" % be for be in zip(bits, errs)))
        print("  pinned anomaly: 2-bit %.4f > 1.58-bit %.4f (Qp=1, max-scaled)" % (e2, e158))

    def test_fakequant_equals_integer_matmul(self):
        """Quantize-dequantize then a float matmul is the same computation as an integer matmul
        with the scales applied afterwards (the BitNet/hardware form): the per-token scale is
        constant along the reduction axis, so it factors straight out of the sum. The two differ
        only in accumulator precision, which is a property of the compute dtype."""
        x = torch.randn(4, IN, dtype=DTYPE)
        W = torch.randn(OUT, IN, dtype=DTYPE)
        bits = 8
        Qp = 2 ** (bits - 1) - 1

        y_fake = torch.nn.functional.linear(fake_quant_activation(x, bits, None), W)

        scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_int = (x * scale).round().clamp(-Qp - 1, Qp)
        self.assertTrue(torch.equal(x_int, x_int.round()), "activations are not integral")
        self.assertLessEqual(x_int.abs().max().item(), Qp + 1, "activations outside int8 range")
        y_int = torch.nn.functional.linear(x_int, W) / scale

        err = rel_err(y_fake, y_int)
        self.assertLess(err, 1e-5, f"fake-quant and integer-matmul paths disagree: {err:.2e}")
        print(f"\n  fake-quant == integer matmul + post-rescale (rel_err={err:.2e})")

    def test_act_config_validated_at_init(self):
        """act_bits/act_group_size are checked once at construction, not on every forward."""
        with self.assertRaises(AssertionError):
            layer(act_bits=5.5)
        with self.assertRaises(AssertionError):  # below MIN_ACT_GROUP_SIZE
            layer(act_bits=8, act_group_size=4)
        with self.assertRaises(AssertionError):  # exceeds the channel dimension
            layer(act_bits=8, act_group_size=IN * 2)
        with self.assertRaises(AssertionError):  # not a divisor of in_features
            layer(act_bits=8, act_group_size=48)
        layer(act_bits=8, act_group_size=IN)  # per-tensor via an explicit group size is fine


class TestQAT(SeededTest):
    def test_trainable_uses_real_hqq_calibration(self):
        """optimize=True must change the cached scale/zero versus plain min/max, otherwise the
        half-quadratic solver is not running at calibration time at all."""
        w_true = make_linear().weight.float()
        on = HQQLinear(make_linear(), cfg(optimize=True), compute_dtype=DTYPE, device=DEV,
                       del_orig=False, trainable=True)
        off = HQQLinear(make_linear(), cfg(optimize=False), compute_dtype=DTYPE, device=DEV,
                        del_orig=False, trainable=True)
        self.assertTrue(hasattr(on, "calib_scale") and hasattr(on, "calib_zero"))
        self.assertFalse(
            torch.allclose(on.calib_scale, off.calib_scale)
            and torch.allclose(on.calib_zero, off.calib_zero),
            "optimize=True produced identical scale/zero to optimize=False",
        )
        on.eval(); off.eval()
        e_on = rel_err(on._weight_calibrated().float(), w_true)
        e_off = rel_err(off._weight_calibrated().float(), w_true)
        self.assertLess(e_on, e_off, f"optimize=True ({e_on:.5f}) not better than False ({e_off:.5f})")
        print(f"\n  HQQ calibration active: rel_err optimize=True {e_on:.5f} < False {e_off:.5f}")

    def test_eval_frozen_train_dynamic(self):
        lay = layer(trainable=True)
        x = torch.randn(4, IN, dtype=DTYPE)
        lay.eval()
        self.assertTrue(torch.equal(lay(x), lay(x)), "eval() forward is not deterministic")

        # Perturb the weights without recalibrating: eval() must keep using the cached scale/zero.
        scale_before = lay.calib_scale.clone()
        with torch.no_grad():
            lay.master_weight.add_(torch.randn_like(lay.master_weight) * 0.05)
        lay(x)
        self.assertTrue(torch.equal(scale_before, lay.calib_scale), "eval() mutated the calibration")

        lay.recalibrate()
        self.assertFalse(torch.equal(scale_before, lay.calib_scale), "recalibrate() did not update")

    def test_gradients_reach_master_weight(self):
        lay = layer(trainable=True)
        lay.train()
        x = torch.randn(8, IN, dtype=DTYPE)
        target = torch.randn(8, OUT, dtype=DTYPE)

        ((lay(x) - target) ** 2).mean().backward()
        self.assertIsNotNone(lay.master_weight.grad, "no grad on the master weight")
        gnorm = lay.master_weight.grad.norm().item()
        self.assertGreater(gnorm, 0, "master weight grad is all zeros")
        self.assertTrue(torch.isfinite(lay.master_weight.grad).all(), "non-finite grads")

        opt = torch.optim.SGD(lay.parameters(), lr=1e-2)
        losses = []
        for _ in range(60):
            opt.zero_grad()
            loss = ((lay(x) - target) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0], f"loss did not decrease: {losses[0]} -> {losses[-1]}")
        print(f"\n  grads flow (|g|={gnorm:.4f}); loss {losses[0]:.4f} -> {losses[-1]:.4f} / 60 steps")

    def test_freeze_matches_fresh(self):
        lay = layer(seed=3, trainable=True)
        lay.freeze()
        self.assertFalse(lay.trainable)
        self.assertIsNotNone(lay.W_q, "freeze() did not pack")
        self.assertFalse(hasattr(lay, "master_weight"), "freeze() left the fp master behind")
        self.assertNotIn("forward", lay.__dict__, "a frozen layer with no act_bits needs no forward")

        fresh = layer(seed=3)
        err = rel_err(lay.dequantize().float(), fresh.dequantize().float())
        self.assertLess(err, 1e-6, f"frozen layer differs from a fresh HQQLinear: {err}")
        lay.eval(); fresh.eval()
        x = torch.randn(4, IN, dtype=DTYPE)
        self.assertLess(rel_err(lay(x), fresh(x)), 1e-6, "frozen forward differs from fresh")

    def test_freeze_after_training(self):
        """The point of the mode: train, then collapse back to a packed HQQLinear whose weights
        are the trained ones."""
        lay = layer(trainable=True)
        w_before = lay.master_weight.detach().clone()
        lay.train()
        x = torch.randn(8, IN, dtype=DTYPE)
        target = torch.randn(8, OUT, dtype=DTYPE)
        opt = torch.optim.SGD(lay.parameters(), lr=1e-2)
        for _ in range(40):
            opt.zero_grad()
            ((lay(x) - target) ** 2).mean().backward()
            opt.step()

        w_trained = lay.master_weight.detach().clone()
        moved = rel_err(w_trained, w_before)
        self.assertGreater(moved, 1e-3, f"training did not move the weights ({moved:.2e})")
        lay.freeze()

        ref_lin = nn.Linear(IN, OUT, bias=True).to(device=DEV, dtype=DTYPE)
        with torch.no_grad():
            ref_lin.weight.copy_(w_trained)
        fresh = HQQLinear(ref_lin, cfg(), compute_dtype=DTYPE, device=DEV, del_orig=False)
        err = rel_err(lay.dequantize().float(), fresh.dequantize().float())
        self.assertLess(err, 1e-6,
                        f"frozen layer does not match a fresh one from trained weights: {err:.2e}")
        self.assertGreater(rel_err(lay.dequantize().float(), w_before.float()), 1e-3,
                           "frozen layer still reflects the pre-training weight")
        print(f"\n  trained (moved {moved:.4f}), froze, matches fresh-from-trained ({err:.1e})")

    def test_trainable_survives_device_move(self):
        """cuda()/to() must work before freeze(). A trainable layer has no packed weight, so
        meta is None and upstream's `self.meta["compute_dtype"] = ...` raised TypeError -
        which is what accelerate's model.to(device) calls, so QAT training could not start."""
        lay = layer(trainable=True)
        self.assertIsNone(lay.meta, "a trainable layer should have no meta before freeze()")
        x = torch.randn(4, IN, dtype=DTYPE)
        lay.eval()
        before = lay(x)
        lay.cuda(DEV)  # the call accelerate makes; must not raise
        self.assertLess(rel_err(lay(x), before), 1e-6, "moving the layer changed the output")
        self.assertTrue(lay.master_weight.requires_grad, "master weight lost requires_grad")

    def test_freeze_keeps_act_quant(self):
        """freeze() flips trainable off, so the forward has to be reinstalled - otherwise the
        layer would keep calling forward_qat with no master weight, or drop act_bits entirely."""
        x = torch.randn(4, IN, dtype=DTYPE)
        lay = layer(seed=3, act_bits=8, trainable=True)
        lay.eval()
        lay.freeze()
        self.assertIn("forward", lay.__dict__, "freeze() dropped the act-quant forward")

        with_act = layer(seed=3, act_bits=8); with_act.eval()
        without_act = layer(seed=3); without_act.eval()
        self.assertLess(rel_err(lay(x), with_act(x)), 1e-6,
                        "frozen layer stopped quantizing activations")
        self.assertGreater(rel_err(lay(x), without_act(x)), 0,
                           "activations are not being quantized")

    def test_survives_transformers_weight_property_patch(self):
        """transformers/quantizers/quantizer_hqq.py sets HQQLinear.weight to a property returning
        an empty tensor. A property is a data descriptor, so it beats instance attribute lookup -
        which is why the fp master is named master_weight."""
        patched = "weight" in HQQLinear.__dict__
        if not patched:
            HQQLinear.weight = property(
                lambda self: torch.empty(0, dtype=self.compute_dtype, device=self.device)
            )
        try:
            self.assertIsInstance(HQQLinear.__dict__.get("weight"), property, "patch not in effect")
            lay = layer(trainable=True)
            self.assertIsInstance(lay.master_weight, nn.Parameter)
            self.assertEqual(lay.master_weight.numel(), IN * OUT, "master weight is not the real tensor")
            self.assertEqual(lay.weight.numel(), 0, "expected the transformers dummy property here")
            lay.train()
            (lay(torch.randn(4, IN, dtype=DTYPE)) ** 2).mean().backward()
            self.assertIsNotNone(lay.master_weight.grad)
            self.assertTrue(torch.isfinite(lay.master_weight.grad).all())
            lay.freeze()
            self.assertIsNotNone(lay.W_q)
            self.assertFalse(hasattr(lay, "master_weight"))
        finally:
            if not patched:
                del HQQLinear.weight


class TestConfig(SeededTest):
    def test_settings_via_quant_config_dict(self):
        """The dict route is what lets an unmodified transformers.HqqConfig carry these."""
        c = cfg()
        c["act_bits"] = 8
        c["trainable"] = True
        lay = HQQLinear(make_linear(), c, compute_dtype=DTYPE, device=DEV, del_orig=False)
        self.assertTrue(lay.trainable, "trainable not picked up from quant_config")
        self.assertEqual(lay.act_bits, 8, "act_bits not picked up from quant_config")
        self.assertTrue(hasattr(lay, "calib_scale"), "trainable path did not initialize")
        # the keys must be consumed, or initialize()'s quantize(**quant_config) would raise
        self.assertNotIn("act_bits", lay.quant_config)
        self.assertNotIn("trainable", lay.quant_config)

        lay.train()
        ((lay(torch.randn(4, IN, dtype=DTYPE))) ** 2).mean().backward()
        self.assertIsNotNone(lay.master_weight.grad)

    def test_kwargs_override_quant_config(self):
        """A kwarg wins over the dict in BOTH directions - trainable=False must be able to turn
        off a config that asks for it, which `bool(trainable or cfg_trainable)` could not do."""
        c_on = cfg()
        c_on["trainable"] = True
        off = HQQLinear(make_linear(), c_on, compute_dtype=DTYPE, device=DEV, del_orig=False,
                        trainable=False)
        self.assertFalse(off.trainable, "trainable=False did not override quant_config")
        self.assertIsNotNone(off.W_q, "the layer should have taken the packed path")
        self.assertFalse(hasattr(off, "master_weight"))

        c_off = cfg()
        c_off["trainable"] = False
        on = HQQLinear(make_linear(), c_off, compute_dtype=DTYPE, device=DEV, del_orig=False,
                       trainable=True)
        self.assertTrue(on.trainable, "trainable=True did not override quant_config")

        plain = layer()
        self.assertFalse(plain.trainable)
        self.assertIsNone(plain.act_bits)

    def test_base_quant_config_carries_act_options(self):
        """BaseQuantizeConfig(nbits=4, act_bits=8) works, and a default config is unchanged."""
        default = BaseQuantizeConfig(nbits=4, group_size=64, axis=1)
        self.assertEqual(
            set(default),
            {"weight_quant_params", "scale_quant_params", "zero_quant_params", "offload_meta"},
            "a default config gained keys - consumers that splat it into a closed signature break",
        )

        c = BaseQuantizeConfig(nbits=4, group_size=GROUP, axis=1, act_bits=8, act_group_size=32,
                               trainable=True)
        self.assertEqual(c["act_bits"], 8)
        self.assertEqual(c["act_group_size"], 32)
        self.assertTrue(c["trainable"])

        lay = HQQLinear(make_linear(), c, compute_dtype=DTYPE, device=DEV, del_orig=False)
        self.assertEqual(lay.act_bits, 8)
        self.assertEqual(lay.act_group_size, 32)
        self.assertTrue(lay.trainable)


class _Model(nn.Module):
    """prepare_for_inference expects an HF-style model: it reads .device/.dtype off it."""

    def __init__(self, **kwargs):
        super().__init__()
        self.proj = layer(**kwargs)
        self.device, self.dtype = DEV, DTYPE

    def forward(self, x):
        return self.proj(x)


class TestInferencePatching(SeededTest):
    """prepare_for_inference replaces each layer's forward, including the act-quantizing one."""

    @staticmethod
    def model(**kwargs):
        return _Model(**kwargs)

    def test_patch_hqq_inference_honours_act_bits(self):
        x = torch.randn(4, IN, dtype=DTYPE)
        m = self.model(act_bits=8)
        m.eval()
        y_before = m(x)
        prepare_for_inference(m)
        y_after = m(x)
        self.assertLess(rel_err(y_after, y_before), 1e-6,
                        "prepare_for_inference changed the numbers - act_bits was dropped")

        plain = self.model()
        plain.eval()
        prepare_for_inference(plain)
        self.assertGreater(rel_err(y_after, plain(x)), 0,
                           "patched act layer matches a patched plain one - nothing is quantized")

    def test_rejects_act_bits_on_external_backend(self):
        x = torch.randn(4, IN, dtype=DTYPE)
        m = self.model(act_bits=8)
        m.eval()
        y_before = m(x)
        with self.assertRaises(RuntimeError) as ctx:
            prepare_for_inference(m, backend="gemlite")
        self.assertIn("act_bits", str(ctx.exception))
        # the guard runs before anything is patched, so the model is left untouched
        self.assertTrue(torch.equal(m(x), y_before), "model was mutated before the guard fired")

    def test_rejects_trainable_layers(self):
        m = self.model(trainable=True)
        m.eval()
        with self.assertRaises(RuntimeError) as ctx:
            prepare_for_inference(m)
        self.assertIn("freeze()", str(ctx.exception))

        m.proj.freeze()
        prepare_for_inference(m)  # fine once frozen


if __name__ == "__main__":
    unittest.main(verbosity=2)
