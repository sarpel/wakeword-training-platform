"""
Benchmark script for Focal Loss and QAT Accuracy Recovery.
Compares:
1. Standard CE Training
2. Focal Loss Training
3. QAT Fine-tuning Accuracy Drop
"""

import torch
import torch.nn as nn
import time
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from src.config.defaults import WakewordConfig
from src.training.trainer import Trainer
from src.training.checkpoint_manager import CheckpointManager
from src.models.architectures import create_model

def create_synthetic_dataset(num_samples=1000, hard_negative_ratio=0.2):
    # 1 channel, 64 mel bins, 50 time steps
    data = torch.randn(num_samples, 1, 64, 50)
    labels = torch.randint(0, 2, (num_samples,))
    
    # Metadata for hard negatives
    is_hard_negative = torch.zeros(num_samples)
    num_hn = int(num_samples * hard_negative_ratio)
    is_hard_negative[:num_hn] = 1.0
    
    # Make hard negatives actually hard (close to decision boundary)
    # For simplicity, we just keep them as they are but marked
    
    metadata = {"is_hard_negative": is_hard_negative}
    
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, data, labels, metadata):
            self.data = data
            self.labels = labels
            self.metadata = metadata
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx], {k: v[idx] for k, v in self.metadata.items()}
            
    return CustomDataset(data, labels, metadata)

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Benchmarking on {device}")
    
    results_dir = Path("logs/benchmarks/focal_qat")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = create_synthetic_dataset()
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32)
    
    # 1. Standard CE
    print("\n--- Running Standard CE Benchmark ---")
    config_ce = WakewordConfig()
    config_ce.loss.loss_function = "cross_entropy"
    config_ce.training.epochs = 5
    
    model_ce = create_model("tiny_conv", num_classes=2)
    trainer_ce = Trainer(model_ce, train_loader, val_loader, config_ce, 
                         CheckpointManager(results_dir/"ce"), device=device)
    res_ce = trainer_ce.train()
    
    # 2. Focal Loss
    print("\n--- Running Focal Loss Benchmark ---")
    config_focal = WakewordConfig()
    config_focal.loss.loss_function = "focal_loss"
    config_focal.loss.focal_gamma = 2.0
    config_focal.training.epochs = 5
    
    model_focal = create_model("tiny_conv", num_classes=2)
    trainer_focal = Trainer(model_focal, train_loader, val_loader, config_focal, 
                            CheckpointManager(results_dir/"focal"), device=device)
    res_focal = trainer_focal.train()
    
    # 3. QAT Recovery
    print("\n--- Running QAT Recovery Benchmark ---")
    config_qat = WakewordConfig()
    config_qat.training.epochs = 6
    config_qat.qat.enabled = True
    config_qat.qat.start_epoch = 3
    
    model_qat = create_model("tiny_conv", num_classes=2)
    trainer_qat = Trainer(model_qat, train_loader, val_loader, config_qat, 
                           CheckpointManager(results_dir/"qat"), device=device)
    res_qat = trainer_qat.train()
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Standard CE Best Val F1: {res_ce['best_val_f1']:.4f}")
    print(f"Focal Loss Best Val F1:  {res_focal['best_val_f1']:.4f}")
    
    if "qat_report" in res_qat:
        report = res_qat["qat_report"]
        print(f"QAT FP32 Acc: {report['fp32_acc']:.4f}")
        print(f"QAT INT8 Acc: {report['quant_acc']:.4f}")
        print(f"Accuracy Drop: {report['drop']*100:.2f}%")
        
        if report['drop'] < 0.02:
            print("✅ Accuracy drop is within limits (< 2%)")
        else:
            print("❌ Accuracy drop exceeds 2%")
    else:
        print("❌ QAT Report not generated")

if __name__ == "__main__":
    run_benchmark()
