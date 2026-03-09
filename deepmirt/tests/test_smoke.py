#!/usr/bin/env python3
"""
End-to-end smoke test — verify that all training pipeline components work correctly.

Usage:
    conda run -n deeplearn python -m pytest deepmirt/tests/test_smoke.py -v --tb=short
    conda run -n deeplearn python -m unittest deepmirt.tests.test_smoke -v
"""

import sys
import unittest
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestSmokeEnvironment(unittest.TestCase):
    """Environment and dependency verification."""

    def test_import_dependencies(self):
        """Verify all ML dependencies can be imported."""

    def test_rnafm_model_loads(self):
        """Verify RNA-FM model can be loaded."""
        import fm
        model, alphabet = fm.pretrained.rna_fm_t12()
        self.assertEqual(len(model.layers), 12)
        self.assertIsNotNone(alphabet.padding_idx)

    def test_gpu_available(self):
        """Verify GPU is available."""
        import torch
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 1)


class TestSmokePreprocessing(unittest.TestCase):
    """Data preprocessing verification."""

    def test_dna_to_rna(self):
        """Verify T->U conversion."""
        from deepmirt.data_module.preprocessing import dna_to_rna
        self.assertEqual(dna_to_rna('ATCGATCG'), 'AUCGAUCG')
        self.assertEqual(dna_to_rna('AUCGAUCG'), 'AUCGAUCG')  # idempotent
        self.assertEqual(dna_to_rna('atcg'), 'AUCG')

    def test_validate_rna_sequence(self):
        """Verify RNA sequence validation."""
        from deepmirt.data_module.preprocessing import validate_rna_sequence
        self.assertTrue(validate_rna_sequence('AUCGAUCG', 5, 30))
        self.assertFalse(validate_rna_sequence('ATCG', 5, 30))  # contains T


class TestSmokeDataset(unittest.TestCase):
    """Dataset and DataModule verification."""

    def test_dataset_loads_sample(self):
        """Verify Dataset can load val.csv."""
        import fm

        from deepmirt.data_module.dataset import MiRNATargetDataset
        _, alphabet = fm.pretrained.rna_fm_t12()
        ds = MiRNATargetDataset('deepmirt/data/training/val.csv', alphabet)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn('mirna_tokens', sample)
        self.assertIn('target_tokens', sample)
        self.assertIn('label', sample)

    def test_datamodule_batch(self):
        """Verify DataModule produces correct batch format."""
        from deepmirt.data_module.datamodule import MiRNATargetDataModule
        dm = MiRNATargetDataModule(
            data_dir='deepmirt/data/training',
            batch_size=4, num_workers=0
        )
        dm.setup('fit')
        batch = next(iter(dm.val_dataloader()))
        self.assertEqual(batch['mirna_tokens'].dim(), 2)
        self.assertEqual(batch['target_tokens'].dim(), 2)
        self.assertEqual(batch['labels'].shape[0], 4)


class TestSmokeModel(unittest.TestCase):
    """Model verification."""

    def test_model_forward_pass(self):
        """Verify model forward pass produces correct output shape."""
        import fm
        import torch

        from deepmirt.model.mirna_target_model import MiRNATargetModel

        model = MiRNATargetModel(freeze_backbone=True)
        model.eval()
        _, alphabet = fm.pretrained.rna_fm_t12()
        bc = alphabet.get_batch_converter()
        _, _, mirna_tokens = bc([('m1', 'UAGCAGCACGUAAAUAUUGGCG'), ('m2', 'UGAGGUAGUAG')])
        _, _, target_tokens = bc([('t1', 'A' * 40), ('t2', 'U' * 40)])
        with torch.no_grad():
            logits = model(mirna_tokens=mirna_tokens, target_tokens=target_tokens)
        self.assertEqual(logits.shape, (2, 1))


class TestSmokeLightning(unittest.TestCase):
    """Lightning training module verification."""

    def test_lightning_module_training_step(self):
        """Verify Lightning module training_step runs without error."""
        import fm
        import torch

        from deepmirt.training.lightning_module import MiRNATargetLitModule

        config = {
            'model': {'freeze_backbone': True, 'cross_attn_heads': 8, 'cross_attn_layers': 2,
                      'classifier_hidden': [256, 64], 'dropout': 0.3},
            'training': {'lr': 1e-4, 'weight_decay': 1e-5, 'scheduler': 'cosine', 'max_epochs': 30}
        }
        lit_model = MiRNATargetLitModule(config)
        _, alphabet = fm.pretrained.rna_fm_t12()
        bc = alphabet.get_batch_converter()
        _, _, mirna_tokens = bc([('m1', 'UAGCAGCACGUAAAUAUUGGCG'), ('m2', 'UGAGGUAG')])
        _, _, target_tokens = bc([('t1', 'A' * 40), ('t2', 'U' * 40)])
        batch = {
            'mirna_tokens': mirna_tokens,
            'target_tokens': target_tokens,
            'labels': torch.tensor([1.0, 0.0]),
            'attention_mask_mirna': torch.ones_like(mirna_tokens),
            'attention_mask_target': torch.ones_like(target_tokens),
        }
        loss = lit_model.training_step(batch, 0)
        self.assertGreater(loss.item(), 0)


if __name__ == '__main__':
    unittest.main()
