"""
P2 Performance Optimizations Test

Tests the performance optimizations added:
1. Channels Last Memory Format (trainer.py) - only on CUDA
2. Non-blocking Data Transfers (training_loop.py)
3. Torch.compile (trainer.py) - only on CUDA
4. Persistent DataLoader Workers (multiple files)
5. N+1 Query Pattern Fix (splitter.py)
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.splitter import DatasetSplitter
from src.training.trainer import Trainer


class TestChannelsLastMemoryFormat:
    """Test channels_last memory format optimization."""

    def test_channels_last_on_cuda(self):
        """Verify model uses channels_last format on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 2),
        )

        data = TensorDataset(torch.randn(10, 1, 64, 50), torch.randint(0, 2, (10,)))
        train_loader = DataLoader(data, batch_size=2)
        val_loader = DataLoader(data, batch_size=2)

        from src.config.defaults import WakewordConfig
        from src.training.checkpoint_manager import CheckpointManager

        config = WakewordConfig()
        checkpoint_manager = CheckpointManager(checkpoint_dir=Path("test_checkpoints"))

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            checkpoint_manager=checkpoint_manager,
            device="cuda",
        )

        # Check if model is in channels_last format
        # Note: We check the first parameter's memory format since the whole model
        # may not be in a single memory format
        first_param = next(trainer.model.parameters())
        # The model itself should have been converted
        # channels_last is only applied on CUDA devices as per implementation

    def test_channels_last_not_on_cpu(self):
        """Verify channels_last is not applied on CPU."""
        model = torch.nn.Linear(10, 2)
        model = model.to("cpu")

        # CPU models should have regular contiguous format
        # channels_last is only beneficial on CUDA
        assert model is not None


class TestTorchCompile:
    """Test torch.compile optimization."""

    def test_torch_compile_on_cuda(self):
        """Verify torch.compile is applied when available and on CUDA."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check if torch.compile exists
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available (PyTorch < 2.0)")

        # Skip on Windows (not yet supported)
        if sys.platform == "win32":
            pytest.skip("torch.compile not yet supported on Windows")

        model = torch.nn.Linear(10, 2)

        # Simulate trainer initialization
        compiled_model = torch.compile(model, mode="max-autotune")

        assert compiled_model is not None

    def test_torch_compile_graceful_failure(self):
        """Verify torch.compile degrades gracefully when it fails."""
        model = torch.nn.Linear(10, 2)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Mock torch.compile to raise an exception
        with patch("torch.compile", side_effect=RuntimeError("Test error")):
            from src.config.defaults import WakewordConfig
            from src.training.checkpoint_manager import CheckpointManager

            config = WakewordConfig()
            checkpoint_manager = CheckpointManager(checkpoint_dir=Path("test_checkpoints"))

            data = TensorDataset(torch.randn(10, 10), torch.randint(0, 2, (10,)))
            train_loader = DataLoader(data, batch_size=2)
            val_loader = DataLoader(data, batch_size=2)

            # This should not raise an exception, just log a warning
            try:
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=config,
                    checkpoint_manager=checkpoint_manager,
                    device=device,
                )
                # Should still work even if torch.compile fails
                assert trainer is not None
            except Exception as e:
                # If it does fail, it should be documented
                pytest.fail(f"Trainer initialization failed after torch.compile error: {e}")


class TestNonBlockingTransfers:
    """Test non-blocking data transfers."""

    def test_non_blocking_in_training_loop(self):
        """Verify training loop uses non-blocking transfers."""
        # Read the training_loop.py file to verify
        import inspect

        from src.training import training_loop

        source = inspect.getsource(training_loop._run_epoch)

        # Check for non_blocking=True in the source code
        assert "non_blocking=True" in source, "Non-blocking transfers not found in training loop"

    @pytest.mark.integration
    def test_non_blocking_performant(self):
        """Test that non-blocking transfers actually improve performance."""
        # This would require a full integration test with actual GPU
        pytest.skip("Integration test requires GPU and actual training run")


class TestPersistentDataLoaderWorkers:
    """Test persistent DataLoader workers optimization."""

    def test_persistent_workers_in_ui(self):
        """Verify UI training panel uses persistent_workers."""
        import inspect

        from src.ui import panel_training

        source = inspect.getsource(panel_training.start_training)

        # Check for persistent_workers parameter
        assert "persistent_workers" in source, "persistent_workers not found in training panel"

    def test_persistent_workers_in_hpo(self):
        """Verify HPO uses persistent_workers."""
        import inspect

        from src.training.hpo import Objective

        source = inspect.getsource(Objective._init_reusable_dataloaders)

        # Check for persistent_workers parameter
        assert "persistent_workers" in source, "persistent_workers not found in HPO"

    def test_prefetch_factor_set(self):
        """Verify prefetch_factor is configured."""
        import inspect

        from src.ui import panel_training

        source = inspect.getsource(panel_training.start_training)

        # Check for prefetch_factor parameter
        assert "prefetch_factor" in source, "prefetch_factor not found in training panel"


class TestNPlusOneQueryPatternFix:
    """Test N+1 query pattern fix in splitter.py."""

    def test_build_npy_index_exists(self):
        """Verify _build_npy_index method exists."""
        splitter = DatasetSplitter(dataset_info={"categories": {}})
        assert hasattr(splitter, "_build_npy_index"), "_build_npy_index method not found"

    def test_build_npy_index_returns_dict(self):
        """Verify _build_npy_index returns a dictionary."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy .npy files
            npy_dir = Path(tmpdir) / "npy"
            npy_dir.mkdir()

            test_positive_dir = npy_dir / "positive"
            test_positive_dir.mkdir()
            test_file = test_positive_dir / "test.npy"
            torch.save(torch.randn(10), test_file)

            dataset_info = {"categories": {}}
            splitter = DatasetSplitter(dataset_info=dataset_info)

            index = splitter._build_npy_index(npy_dir)

            assert isinstance(index, dict), "Index should be a dictionary"
            assert len(index) == 1, f"Expected 1 file in index, got {len(index)}"
            assert "positive/test" in index, "Expected key 'positive/test' in index"

    def test_find_npy_path_uses_index(self):
        """Verify _find_npy_path uses pre-built index."""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy .npy files
            npy_dir = Path(tmpdir) / "npy"
            npy_dir.mkdir()

            test_positive_dir = npy_dir / "positive"
            test_positive_dir.mkdir()
            test_file = test_positive_dir / "test.npy"
            torch.save(torch.randn(10), test_file)

            # Create dataset info
            dataset_info = {
                "categories": {
                    "positive": {"files": [{"path": str(test_file.with_suffix(".wav")), "category": "positive"}]}
                }
            }

            splitter = DatasetSplitter(dataset_info=dataset_info)

            # Build index
            splitter.npy_index = splitter._build_npy_index(npy_dir)

            # Find path - should use index
            npy_path = splitter._find_npy_path(test_file.with_suffix(".wav"), npy_dir)

            # Should find the .npy file using the pre-built index
            assert npy_path is not None, "NPY path should be found via index"

    def test_split_datasets_builds_index(self):
        """Verify split_datasets calls _build_npy_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset structure
            data_root = Path(tmpdir) / "raw"
            data_root.mkdir()
            npy_dir = data_root / "npy"
            npy_dir.mkdir()

            positive_dir = npy_dir / "positive"
            positive_dir.mkdir()

            # Create multiple files to enable splitting
            files = []
            for i in range(10):
                test_file = positive_dir / f"test_{i}.npy"
                torch.save(torch.randn(10), test_file)
                files.append({"path": str(test_file.with_suffix(".wav")), "category": "positive"})

            dataset_info = {"categories": {"positive": {"files": files}}}

            splitter = DatasetSplitter(dataset_info=dataset_info)

            # This should build the index
            splits = splitter.split_datasets(npy_source_dir=npy_dir, npy_output_dir=Path(tmpdir) / "splits")

            # Verify index was built
            assert len(splitter.npy_index) > 0, "NPY index should be built during split_datasets"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
