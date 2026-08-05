# SPDX-License-Identifier: Apache-2.0
"""Reloading a 1.58-bit checkpoint used to lose the bit-width.

`_META_TYPE` types `nbits` as `int`, so `decode_safetensor_type` truncated the stored 1.58 to 1.
The weights were unaffected — unpacking keys off `meta["packing"]`, and scale/zero are stored
directly — so outputs matched and the bug was invisible in any forward-pass check. What was
wrong was `meta["nbits"]` and `quant_config["weight_quant_params"]["nbits"]`, which is what bit
accounting, `freeze()` and a subsequent re-save read.

Run from the repo root:

    python -m unittest tests.test_nbits_reload -v

TestWhisperTiny downloads openai/whisper-tiny and needs transformers; the rest is CPU-only with
no download.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

import hqq as _hqq
from hqq.core.quantize import HQQLinear, BaseQuantizeConfig, Quantizer
from hqq.core.utils import decode_safetensor_type, encode_safetensor_type

_REPO_ROOT = Path(__file__).resolve().parents[1]
assert Path(_hqq.__file__).resolve().is_relative_to(_REPO_ROOT), (
    "importing the wrong hqq: %s (expected under %s)" % (_hqq.__file__, _REPO_ROOT)
)

IN, OUT, GROUP = 128, 64, 64
DEV, DTYPE = "cpu", torch.float32


def make_layer(nbits, seed=0):
    torch.manual_seed(seed)
    lin = nn.Linear(IN, OUT).to(device=DEV, dtype=DTYPE)
    return HQQLinear(lin, BaseQuantizeConfig(nbits=nbits, group_size=GROUP, axis=1),
                     compute_dtype=DTYPE, device=DEV, del_orig=False)


def reload_from(layer):
    fresh = HQQLinear(None, BaseQuantizeConfig(nbits=4, group_size=GROUP, axis=1),
                      compute_dtype=DTYPE, device=DEV)
    fresh.load_state_dict(layer.state_dict())
    return fresh


class TestNbitsReload(unittest.TestCase):
    def test_cause_is_the_int_decode(self):
        """The negative half: the underlying trap is still there, so the fix is load-bearing.
        Encoding is fine — 1.58 is stored as float32 — it is decoding it as int that truncates."""
        stored = encode_safetensor_type(1.58)
        self.assertEqual(stored.dtype, torch.float32, "1.58 is not stored as a float")
        self.assertEqual(decode_safetensor_type(stored, int), 1, "the int decode no longer truncates")
        self.assertAlmostEqual(decode_safetensor_type(stored, float), 1.58, places=6)
        # and this is the value _META_TYPE would have used
        from hqq.core.quantize import _META_TYPE
        self.assertIs(_META_TYPE["nbits"], int, "_META_TYPE changed; this test needs updating")
        print("\n  cause confirmed: stored float32, decoded as int -> 1")

    def test_nbits_survives_reload(self):
        for nbits in (1, 1.58, 2, 3, 4, 8):
            src = make_layer(nbits)
            fresh = reload_from(src)
            self.assertEqual(fresh.meta["nbits"], nbits, f"meta nbits wrong for {nbits}")
            self.assertEqual(fresh.quant_config["weight_quant_params"]["nbits"], nbits,
                             f"quant_config nbits wrong for {nbits}")
            # the value must remain usable by the machinery that indexes on it
            self.assertIn(nbits, Quantizer.SUPPORTED_BITS)
            self.assertEqual(Quantizer.bit_to_packing[fresh.meta["nbits"]], src.meta["packing"])
        print("  nbits survives reload for 1, 1.58, 2, 3, 4, 8")

    def test_nbits_survives_a_second_save(self):
        """The original bug was self-propagating: a reloaded layer wrote the truncated value
        back out, so the corruption became permanent after one save/load cycle."""
        src = make_layer(1.58)
        once = reload_from(src)
        twice = reload_from(once)
        self.assertEqual(twice.meta["nbits"], 1.58, "nbits lost on the second round trip")
        self.assertEqual(once.state_dict()["nbits"].item(), src.state_dict()["nbits"].item())
        print("  nbits survives a re-save (the bug used to become permanent)")

    def test_weights_were_never_the_problem(self):
        """Control: the packed weights and the dequantized values always round-tripped. This is
        why a forward-pass comparison could not detect the bug."""
        x = torch.randn(4, IN, dtype=DTYPE)
        for nbits in (1.58, 4):
            src = make_layer(nbits)
            fresh = reload_from(src)
            src.eval(); fresh.eval()
            self.assertEqual(fresh.meta["packing"], src.meta["packing"])
            self.assertTrue(torch.equal(src.dequantize(), fresh.dequantize()))
            self.assertTrue(torch.equal(src(x), fresh(x)))
        print("  control: packing, dequantize() and outputs were always correct")


class TestWhisperTiny(unittest.TestCase):
    """End-to-end through transformers, at 1.58 bits."""

    MODEL_ID = "openai/whisper-tiny"

    @classmethod
    def setUpClass(cls):
        try:
            from transformers import HqqConfig, WhisperForConditionalGeneration
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest("transformers not available: %s" % e)
        cls.HqqConfig, cls.Whisper = HqqConfig, WhisperForConditionalGeneration

    def test_whisper_tiny_1p58_roundtrip(self):
        torch.manual_seed(0)
        feats = torch.randn(1, 80, 3000, dtype=DTYPE)
        dec = torch.tensor([[50258, 50259, 50359]])

        # proj_out shares its weight with embed_tokens; quantizing it breaks the tie on save.
        q = self.HqqConfig(nbits=1.58, group_size=64, axis=1,
                           skip_modules=["lm_head", "proj_out"])
        model = self.Whisper.from_pretrained(self.MODEL_ID, dtype=DTYPE, device_map=DEV,
                                             quantization_config=q)
        model.eval()
        layers = [m for m in model.modules() if isinstance(m, HQQLinear)]
        self.assertTrue(layers, "no HQQLinear layers were created")
        self.assertEqual(layers[0].meta["nbits"], 1.58)
        with torch.no_grad():
            y_ref = model(input_features=feats, decoder_input_ids=dec).logits

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            back = self.Whisper.from_pretrained(d, dtype=DTYPE, device_map=DEV)
            back.eval()
            got = [m for m in back.modules() if isinstance(m, HQQLinear)]
            self.assertTrue(got, "reloaded model has no HQQLinear layers")
            wrong = [m.meta["nbits"] for m in got if m.meta["nbits"] != 1.58]
            self.assertFalse(wrong, "reloaded nbits is %s, expected 1.58" % set(wrong))
            with torch.no_grad():
                y = back(input_features=feats, decoder_input_ids=dec).logits

        err = ((y - y_ref).norm() / y_ref.norm()).item()
        self.assertLess(err, 1e-6, "reloaded model diverged: %.2e" % err)
        print("\n  whisper-tiny W1.58: %d layers, nbits=1.58 preserved, rel diff %.2e"
              % (len(got), err))


if __name__ == "__main__":
    unittest.main(verbosity=2)
