#!/usr/bin/env python3
"""
端到端 Smoke Test — 验证训练管线各组件正常工作。

运行方式:
    conda run -n deeplearn python -m pytest insect_mirna_target/tests/test_smoke.py -v --tb=short
    conda run -n deeplearn python -m unittest insect_mirna_target.tests.test_smoke -v
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class TestSmokeEnvironment(unittest.TestCase):
    """环境和依赖验证"""

    def test_import_dependencies(self):
        """验证所有 ML 依赖可导入"""
        import fm
        import pytorch_lightning
        import torchmetrics
        import yaml
        import sklearn
        import torch

    def test_rnafm_model_loads(self):
        """验证 RNA-FM 模型可加载"""
        import fm
        model, alphabet = fm.pretrained.rna_fm_t12()
        self.assertEqual(len(model.layers), 12)
        self.assertIsNotNone(alphabet.padding_idx)

    def test_gpu_available(self):
        """验证 GPU 可用"""
        import torch
        self.assertTrue(torch.cuda.is_available())
        self.assertGreaterEqual(torch.cuda.device_count(), 1)


class TestSmokePreprocessing(unittest.TestCase):
    """数据预处理验证"""

    def test_dna_to_rna(self):
        """验证 T→U 转换"""
        from insect_mirna_target.data_module.preprocessing import dna_to_rna
        self.assertEqual(dna_to_rna('ATCGATCG'), 'AUCGAUCG')
        self.assertEqual(dna_to_rna('AUCGAUCG'), 'AUCGAUCG')  # idempotent
        self.assertEqual(dna_to_rna('atcg'), 'AUCG')

    def test_validate_rna_sequence(self):
        """验证 RNA 序列验证"""
        from insect_mirna_target.data_module.preprocessing import validate_rna_sequence
        self.assertTrue(validate_rna_sequence('AUCGAUCG', 5, 30))
        self.assertFalse(validate_rna_sequence('ATCG', 5, 30))  # contains T


class TestSmokeDataset(unittest.TestCase):
    """Dataset 和 DataModule 验证"""

    def test_dataset_loads_sample(self):
        """验证 Dataset 可加载 val.csv"""
        import fm
        from insect_mirna_target.data_module.dataset import MiRNATargetDataset
        _, alphabet = fm.pretrained.rna_fm_t12()
        ds = MiRNATargetDataset('insect_mirna_target/data/training/val.csv', alphabet)
        self.assertGreater(len(ds), 0)
        sample = ds[0]
        self.assertIn('mirna_tokens', sample)
        self.assertIn('target_tokens', sample)
        self.assertIn('label', sample)

    def test_datamodule_batch(self):
        """验证 DataModule 生成正确的 batch"""
        from insect_mirna_target.data_module.datamodule import MiRNATargetDataModule
        dm = MiRNATargetDataModule(
            data_dir='insect_mirna_target/data/training',
            batch_size=4, num_workers=0
        )
        dm.setup('fit')
        batch = next(iter(dm.val_dataloader()))
        self.assertEqual(batch['mirna_tokens'].dim(), 2)
        self.assertEqual(batch['target_tokens'].dim(), 2)
        self.assertEqual(batch['labels'].shape[0], 4)


class TestSmokeModel(unittest.TestCase):
    """模型验证"""

    def test_model_forward_pass(self):
        """验证模型前向传播 shape 正确"""
        import torch
        import fm
        from insect_mirna_target.model.mirna_target_model import MiRNATargetModel

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
    """Lightning 训练模块验证"""

    def test_lightning_module_training_step(self):
        """验证 Lightning module training_step 不报错"""
        import torch
        import fm
        from insect_mirna_target.training.lightning_module import MiRNATargetLitModule

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
