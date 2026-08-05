# SPDX-License-Identifier: Apache-2.0
"""End-to-end check that activation quantization survives a transformers save/reload.

Kept out of test_hqqlinear_qat.py because it downloads openai/whisper-tiny and needs
transformers; that suite is deliberately CPU-only with no model download. Run from the repo
root with this checkout on the path:

    python -m unittest tests.test_act_quant_save_reload -v
"""
import tempfile
import unittest
from pathlib import Path

import torch

import hqq as _hqq
from hqq.core.quantize import HQQLinear
from hqq.utils.patching import restore_act_quant

_REPO_ROOT = Path(__file__).resolve().parents[1]
assert Path(_hqq.__file__).resolve().is_relative_to(_REPO_ROOT), (
    "importing the wrong hqq: %s (expected under %s)" % (_hqq.__file__, _REPO_ROOT)
)

MODEL_ID = "openai/whisper-tiny"
NBITS, GROUP = 1.58, 64        # weights
ACT_BITS, ACT_GROUP = 8, 16    # activations
DEV, DTYPE = "cpu", torch.float32


def _inputs():
    torch.manual_seed(0)
    return (torch.randn(1, 80, 3000, dtype=DTYPE),
            torch.tensor([[50258, 50259, 50359]]))


class TestActQuantSaveReload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import HqqConfig, WhisperForConditionalGeneration
        except ImportError as e:  # pragma: no cover
            raise unittest.SkipTest("transformers not available: %s" % e)
        cls.HqqConfig = HqqConfig
        cls.Whisper = WhisperForConditionalGeneration

    def _load_quantized(self):
        q = self.HqqConfig(nbits=NBITS, group_size=GROUP, axis=1,
                           # proj_out shares its weight with embed_tokens; quantizing it breaks
                           # the tie and the shared-tensor save.
                           skip_modules=["lm_head", "proj_out"])
        q.quant_config["act_bits"] = ACT_BITS
        q.quant_config["act_group_size"] = ACT_GROUP
        return self.Whisper.from_pretrained(
            MODEL_ID, dtype=DTYPE, device_map=DEV, quantization_config=q)

    @staticmethod
    def _act_layers(model):
        return {n: l for n, l in model.named_modules() if isinstance(l, HQQLinear)}

    def test_whisper_tiny_roundtrip(self):
        feats, dec = _inputs()

        model = self._load_quantized()
        model.eval()
        layers = self._act_layers(model)
        self.assertTrue(layers, "no HQQLinear layers were created")
        some = next(iter(layers.values()))
        self.assertEqual(some.act_bits, ACT_BITS)
        self.assertEqual(some.act_group_size, ACT_GROUP)
        with torch.no_grad():
            y_ref = model(input_features=feats, decoder_input_ids=dec).logits
        print("\n  quantized in memory: %d HQQLinear layers, W%s/A%s g%s"
              % (len(layers), NBITS, ACT_BITS, ACT_GROUP))

        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            reloaded = self.Whisper.from_pretrained(d, dtype=DTYPE, device_map=DEV)
            reloaded.eval()

            back = self._act_layers(reloaded)
            self.assertTrue(back, "reloaded model has no HQQLinear layers")

            with torch.no_grad():
                y_raw = reloaded(input_features=feats, decoder_input_ids=dec).logits
            raw_lost = any(l.act_bits is None for l in back.values())
            print("  after from_pretrained: act_bits lost = %s, rel diff vs ref = %.2e"
                  % (raw_lost, self._rel(y_raw, y_ref)))

            restore_act_quant(reloaded)
            for n, l in self._act_layers(reloaded).items():
                self.assertEqual(l.act_bits, ACT_BITS, "act_bits not restored on %s" % n)
                self.assertEqual(l.act_group_size, ACT_GROUP)
                self.assertIn("forward", l.__dict__, "forward not installed on %s" % n)

            with torch.no_grad():
                y_restored = reloaded(input_features=feats, decoder_input_ids=dec).logits

        err = self._rel(y_restored, y_ref)
        print("  after restore_act_quant: rel diff vs ref = %.2e" % err)
        self.assertLess(err, 1e-6, "restored model does not match the pre-save model: %.2e" % err)

    @staticmethod
    def _rel(a, b):
        return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


if __name__ == "__main__":
    unittest.main(verbosity=2)
