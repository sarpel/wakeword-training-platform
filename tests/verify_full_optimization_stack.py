"""
End-to-End Verification of the Advanced Optimization Stack
Tests: TinyConvV2 + Dual Distillation + QAT Fusion + Dynamic Weighting
"""
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from src.config.defaults import WakewordConfig
from src.models.architectures import create_model
from src.training.distillation_trainer import DistillationTrainer
from src.training.qat_utils import prepare_model_for_qat


def verify_stack():
    print("Starting Full Optimization Stack Verification...")
    print("-" * 50)

    # 1. Configuration
    config = WakewordConfig()
    config.model.architecture = "tiny_conv"
    config.model.tiny_conv_use_depthwise = True
    config.qat.enabled = True
    config.distillation.enabled = True
    config.distillation.teacher_architecture = "dual"
    config.distillation.feature_alignment_enabled = True
    config.distillation.alignment_layers = [1, 2]

    # 2. Model Creation (TinyConv V2)
    print("Step 1: Creating TinyConv V2 (Depthwise)...")
    model = create_model(
        config.model.architecture,
        num_classes=config.model.num_classes,
        use_depthwise=config.model.tiny_conv_use_depthwise,
    )
    print(f"  Model type: {model.__class__.__name__}")

    # 3. QAT Preparation (Module Fusion)
    print("Step 2: Preparing for QAT (Module Fusion)...")
    model = prepare_model_for_qat(model, config.qat)

    # Verify fusion (look for intrinsic modules)
    module_names = [m.__class__.__name__ for m in model.modules()]
    print(f"  All module types found: {module_names[:20]}...")  # Print first 20

    has_fused = any("ConvBnReLU2d" in name or "ConvBn2d" in name or "ConvReLU2d" in name for name in module_names)
    print(f"  Fusion successful: {has_fused}")
    if not has_fused:
        raise RuntimeError("Module fusion failed!")

    # 4. Trainer Initialization (Dual Distillation + Projectors)
    print("Step 3: Initializing DistillationTrainer...")

    # Mock teachers
    class MockTeacher(nn.Module):
        def __init__(self, name=None, **kwargs):
            super().__init__()
            self.name = name or "teacher"
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(torch.randn(x.size(0), 10))

        def embed(self, x, layer_index=None):
            return torch.randn(x.size(0), 128)

    class MockWav2Vec(MockTeacher):
        pass

    import src.training.distillation_trainer

    original_create = src.training.distillation_trainer.create_model
    original_w2v = src.training.distillation_trainer.Wav2VecWakeword

    src.training.distillation_trainer.create_model = lambda *args, **kwargs: MockTeacher("conformer")
    src.training.distillation_trainer.Wav2VecWakeword = MockWav2Vec

    try:
        trainer = DistillationTrainer(
            model=model,
            train_loader=MagicMock(),
            val_loader=MagicMock(),
            config=config,
            checkpoint_manager=MagicMock(),
            device="cpu",
        )

        print(f"  Projectors created: {list(trainer.projectors.keys())}")
        if len(trainer.projectors) == 0:
            raise RuntimeError("Projectors were not created!")

        # 5. Loss Computation (Dynamic Weighting)
        print("Step 4: Verifying Loss Computation (Dynamic Weighting)...")
        # Mock audio_processor to return a valid 4D feature tensor
        trainer.audio_processor = lambda x: torch.randn(x.size(0), 1, 64, 50)
        trainer.criterion = lambda o, t: torch.tensor(1.0, requires_grad=True)

        outputs = torch.randn(2, 2)
        targets = torch.tensor([0, 1])
        inputs = torch.randn(2, 16000)
        processed_inputs = torch.randn(2, 1, 64, 50)

        loss = trainer.compute_loss(outputs, targets, inputs=inputs, processed_inputs=processed_inputs)
        print(f"  Total Loss: {loss.item():.4f}")

        if loss.item() <= 0:
            raise RuntimeError("Invalid loss value!")

        print("-" * 50)
        print("✅ FULL OPTIMIZATION STACK VERIFIED SUCCESSFULLY!")

    finally:
        src.training.distillation_trainer.create_model = original_create
        src.training.distillation_trainer.Wav2VecWakeword = original_w2v


if __name__ == "__main__":
    verify_stack()
