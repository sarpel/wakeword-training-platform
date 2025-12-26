#!/usr/bin/env python3
"""
Save HPO Best Parameters as Profiles
This script saves the best hyperparameters from HPO as individual profile JSON files.
"""

import json
from datetime import datetime
from pathlib import Path

# Best HPO parameters from the training run
BEST_HPO_PARAMS = {
    "learning_rate": 0.005139763037197042,
    "weight_decay": 2.9386834150072167e-06,
    "optimizer": "adamw",
    "batch_size": 32,
    "dropout": 0.16729470478503433,
    "background_noise_prob": 0.8988690796739087,
    "rir_prob": 0.4366581080487087,
    "time_stretch_min": 0.9377698892553146,
    "time_stretch_max": 1.172831982504407,
    "freq_mask_param": 12,
    "time_mask_param": 26,
    "loss_function": "cross_entropy",
}

# Best F1 score achieved
BEST_F1_SCORE = 0.971195391262602

# Profile metadata
PROFILE_METADATA = {
    "name": "HPO_Best_2025-12-19",
    "description": f"Best hyperparameters from HPO study with F1 score: {BEST_F1_SCORE}",
    "date_created": datetime.now().isoformat(),
    "f1_score": BEST_F1_SCORE,
    "source": "Optuna HPO Study - 50 trials",
}


def create_training_profile() -> dict:
    """Create Training parameter profile"""
    return {
        "metadata": {
            **PROFILE_METADATA,
            "group": "Training",
            "description": "Training and optimizer hyperparameters from HPO",
        },
        "parameters": {
            "training": {
                "batch_size": BEST_HPO_PARAMS["batch_size"],
                "learning_rate": BEST_HPO_PARAMS["learning_rate"],
            },
            "optimizer": {"optimizer": BEST_HPO_PARAMS["optimizer"], "weight_decay": BEST_HPO_PARAMS["weight_decay"]},
        },
    }


def create_model_profile() -> dict:
    """Create Model parameter profile"""
    return {
        "metadata": {
            **PROFILE_METADATA,
            "group": "Model",
            "description": "Model architecture hyperparameters from HPO",
        },
        "parameters": {"model": {"dropout": BEST_HPO_PARAMS["dropout"]}},
    }


def create_augmentation_profile() -> dict:
    """Create Augmentation parameter profile"""
    return {
        "metadata": {
            **PROFILE_METADATA,
            "group": "Augmentation",
            "description": "Data augmentation hyperparameters from HPO",
        },
        "parameters": {
            "augmentation": {
                "background_noise_prob": BEST_HPO_PARAMS["background_noise_prob"],
                "rir_prob": BEST_HPO_PARAMS["rir_prob"],
                "time_stretch_min": BEST_HPO_PARAMS["time_stretch_min"],
                "time_stretch_max": BEST_HPO_PARAMS["time_stretch_max"],
                "freq_mask_param": BEST_HPO_PARAMS["freq_mask_param"],
                "time_mask_param": BEST_HPO_PARAMS["time_mask_param"],
            }
        },
    }


def create_loss_profile() -> dict:
    """Create Loss function parameter profile"""
    return {
        "metadata": {**PROFILE_METADATA, "group": "Loss", "description": "Loss function configuration from HPO"},
        "parameters": {"loss": {"loss_function": BEST_HPO_PARAMS["loss_function"]}},
    }


def create_complete_profile() -> dict:
    """Create complete configuration profile with all HPO parameters"""
    return {
        "metadata": {
            **PROFILE_METADATA,
            "group": "Complete",
            "description": f"Complete HPO configuration - F1: {BEST_F1_SCORE}",
        },
        "parameters": {
            "training": {
                "batch_size": BEST_HPO_PARAMS["batch_size"],
                "learning_rate": BEST_HPO_PARAMS["learning_rate"],
            },
            "optimizer": {"optimizer": BEST_HPO_PARAMS["optimizer"], "weight_decay": BEST_HPO_PARAMS["weight_decay"]},
            "model": {"dropout": BEST_HPO_PARAMS["dropout"]},
            "augmentation": {
                "background_noise_prob": BEST_HPO_PARAMS["background_noise_prob"],
                "rir_prob": BEST_HPO_PARAMS["rir_prob"],
                "time_stretch_min": BEST_HPO_PARAMS["time_stretch_min"],
                "time_stretch_max": BEST_HPO_PARAMS["time_stretch_max"],
                "freq_mask_param": BEST_HPO_PARAMS["freq_mask_param"],
                "time_mask_param": BEST_HPO_PARAMS["time_mask_param"],
            },
            "loss": {"loss_function": BEST_HPO_PARAMS["loss_function"]},
        },
    }


def save_profiles():
    """Save all HPO parameter profiles to JSON files"""
    # Create profiles directory
    profiles_dir = Path("configs/profiles")
    profiles_dir.mkdir(parents=True, exist_ok=True)

    # Define profiles to save
    profiles = {
        "hpo_best_training.json": create_training_profile(),
        "hpo_best_model.json": create_model_profile(),
        "hpo_best_augmentation.json": create_augmentation_profile(),
        "hpo_best_loss.json": create_loss_profile(),
        "hpo_best_complete.json": create_complete_profile(),
    }

    # Save each profile
    saved_files = []
    for filename, profile_data in profiles.items():
        filepath = profiles_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        saved_files.append(filepath)
        print(f"[OK] Saved: {filepath}")

    # Create a summary file
    summary = {
        "hpo_run_date": PROFILE_METADATA["date_created"],
        "best_f1_score": BEST_F1_SCORE,
        "best_parameters": BEST_HPO_PARAMS,
        "saved_profiles": [str(f) for f in saved_files],
        "usage": {
            "description": "Load these profiles in the Gradio UI to apply HPO results",
            "profiles": {
                "Training": "configs/profiles/hpo_best_training.json",
                "Model": "configs/profiles/hpo_best_model.json",
                "Augmentation": "configs/profiles/hpo_best_augmentation.json",
                "Loss": "configs/profiles/hpo_best_loss.json",
                "Complete": "configs/profiles/hpo_best_complete.json",
            },
        },
    }

    summary_path = profiles_dir / "hpo_best_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved summary: {summary_path}")

    print("\n" + "=" * 60)
    print("SUCCESS: HPO Best Parameters Successfully Saved!")
    print("=" * 60)
    print(f"\nBest F1 Score: {BEST_F1_SCORE}")
    print(f"Profiles saved to: {profiles_dir.absolute()}")
    print("\nSaved profiles:")
    for filepath in saved_files:
        print(f"  - {filepath.name}")
    print(f"  - {summary_path.name}")

    return saved_files


if __name__ == "__main__":
    save_profiles()
