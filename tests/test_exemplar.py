# SPDX-License-Identifier: Apache-2.0
"""The testing bar for this repo (see CLAUDE.md): nine flat tests, one behavior each,
arrange-act-assert inline, explicit tolerances, CPU-only, sub-second. Run from the repo root:

    python -m unittest tests.test_exemplar -v
"""
import unittest
from pathlib import Path

import torch
from torch import nn

import hqq as _hqq
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig

# hqq is also pip-installed; without this the suite would silently test the wrong code.
assert Path(_hqq.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1]), (
    "importing the wrong hqq: %s - run from the repo root" % _hqq.__file__
)

IN, OUT, GROUP = 32, 8, 16
CFG = dict(compute_dtype=torch.float32, device="cpu", del_orig=False)


def cfg(nbits=4):
    return BaseQuantizeConfig(nbits=nbits, group_size=GROUP, axis=1)


class TestExemplar(unittest.TestCase):
    def test_forward_output_shape_dtype_and_no_nans(self):
        torch.manual_seed(0)
        layer = HQQLinear(nn.Linear(IN, OUT), cfg(), **CFG)
        y = layer(torch.randn(2, IN))
        self.assertEqual(y.shape, (2, OUT))
        self.assertEqual(y.dtype, torch.float32)
        self.assertFalse(torch.isnan(y).any(), "forward produced NaNs")

    def test_forward_matches_explicit_dequantized_linear(self):
        torch.manual_seed(1)
        layer = HQQLinear(nn.Linear(IN, OUT), cfg(), **CFG)
        x = torch.randn(2, IN)
        reference = torch.nn.functional.linear(x, layer.dequantize(), layer.bias)
        # Same math via a different code path; only float accumulation order differs.
        torch.testing.assert_close(layer(x), reference, rtol=1e-5, atol=1e-6)

    def test_quant_error_decreases_as_bits_increase(self):
        torch.manual_seed(2)
        lin = nn.Linear(IN, OUT)
        errs = []
        for nbits in (2, 4, 8):
            layer = HQQLinear(lin, cfg(nbits), **CFG)
            err = (layer.dequantize() - lin.weight).norm() / lin.weight.norm()
            errs.append(err.item())
        self.assertGreater(errs[0], errs[1], "4 bits should beat 2")
        self.assertGreater(errs[1], errs[2], "8 bits should beat 4")

    def test_gradients_reach_exactly_the_trainable_params(self):
        torch.manual_seed(3)
        # del_orig left True here so the retired nn.Linear's params don't muddy the census.
        layer = HQQLinear(nn.Linear(IN, OUT), cfg(), trainable=True,
                          compute_dtype=torch.float32, device="cpu")
        layer(torch.randn(2, IN)).sum().backward()
        names = {n for n, _ in layer.named_parameters()}
        self.assertEqual(names, {"master_weight", "bias"})  # everything else is frozen
        for name, p in layer.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} got no gradient")
            self.assertGreater(p.grad.pow(2).sum().item(), 0, f"{name} gradient is zero")

    def test_batch_independence(self):
        torch.manual_seed(4)
        layer = HQQLinear(nn.Linear(IN, OUT), cfg(), **CFG)
        x = torch.randn(2, IN, requires_grad=True)
        layer(x)[0].sum().backward()  # loss from sample 0 only
        self.assertGreater(x.grad[0].abs().sum().item(), 0)
        torch.testing.assert_close(x.grad[1], torch.zeros(IN), rtol=0, atol=0)

    def test_single_optimizer_step_reduces_loss(self):
        torch.manual_seed(5)
        layer = HQQLinear(nn.Linear(IN, OUT), cfg(), trainable=True, **CFG)
        x, target = torch.randn(2, IN), torch.zeros(2, OUT)
        opt = torch.optim.SGD(layer.parameters(), lr=1e-2)
        before = torch.nn.functional.mse_loss(layer(x), target)
        before.backward()
        opt.step()
        after = torch.nn.functional.mse_loss(layer(x), target)
        self.assertLess(after.item(), before.item())

    def test_save_reload_roundtrip_outputs_identical(self):
        torch.manual_seed(6)
        layer = HQQLinear(nn.Linear(IN, OUT), cfg(), act_bits=8, **CFG)
        fresh = HQQLinear(None, cfg(), compute_dtype=torch.float32, device="cpu")
        fresh.load_state_dict(layer.state_dict())
        x = torch.randn(2, IN)
        # Bit-identical: the reloaded layer must be the same layer, not an approximation.
        torch.testing.assert_close(fresh(x), layer(x), rtol=0, atol=0)

    def test_freeze_matches_fresh_quantized_layer(self):
        torch.manual_seed(7)
        lin = nn.Linear(IN, OUT)
        frozen = HQQLinear(lin, cfg(), trainable=True, **CFG).freeze()
        fresh = HQQLinear(lin, cfg(), **CFG)
        x = torch.randn(2, IN)
        # Both sides run the same HQQ calibration on the same weights; only float noise differs.
        torch.testing.assert_close(frozen(x), fresh(x), rtol=1e-5, atol=1e-6)

    def test_inference_deterministic_given_seed(self):
        outs = []
        for _ in range(2):
            torch.manual_seed(8)
            layer = HQQLinear(nn.Linear(IN, OUT), cfg(), **CFG)
            outs.append(layer(torch.randn(2, IN)))
        torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
