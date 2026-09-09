"""Regression checks for patched model copies used during EMA validation."""

import copy
import unittest

import torch
from torch import nn

from train_words_3000 import patch_forward_no_final_logit_norm
from utils.utils import ModelEma


class TinyRecognizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm = nn.Identity()
        self.patch_embed = nn.BatchNorm2d(1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, 1))
        self.blocks = nn.ModuleList()
        self.norm = nn.Identity()
        self.head = nn.Linear(1, 1, bias=False)


class EmaForwardTests(unittest.TestCase):
    def test_copy_uses_its_own_parameters_and_mode(self):
        model = TinyRecognizer()
        patch_forward_no_final_logit_norm(model)
        model.eval()
        with torch.no_grad():
            model.head.weight.fill_(2)
        copied = copy.deepcopy(model)
        with torch.no_grad():
            copied.head.weight.fill_(7)
        inputs = torch.ones(1, 1, 1, 2)
        self.assertIs(copied.forward.__self__, copied)
        torch.testing.assert_close(copied(inputs), model(inputs) * 3.5)

    def test_ema_validation_does_not_touch_training_batchnorm(self):
        model = TinyRecognizer()
        patch_forward_no_final_logit_norm(model)
        model.train()
        ema = ModelEma(model, decay=0.5)
        before = model.patch_embed.num_batches_tracked.clone()
        ema.ema(torch.ones(1, 1, 1, 2))
        self.assertTrue(model.training)
        self.assertFalse(ema.ema.training)
        torch.testing.assert_close(model.patch_embed.num_batches_tracked, before)

    def test_ema_update_changes_ema_predictions(self):
        model = TinyRecognizer()
        patch_forward_no_final_logit_norm(model)
        model.eval()
        with torch.no_grad():
            model.head.weight.fill_(2)
        ema = ModelEma(model, decay=0.5)
        with torch.no_grad():
            model.head.weight.fill_(6)
        ema.update(model)
        inputs = torch.ones(1, 1, 1, 2)
        torch.testing.assert_close(ema.ema(inputs), model(inputs) * (4 / 6))


if __name__ == "__main__":
    unittest.main()
