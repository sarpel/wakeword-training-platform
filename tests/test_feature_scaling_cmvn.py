"""
Tests for CMVN normalization and feature scaling in FeatureExtractor.
"""
import pytest
import torch
import numpy as np
from src.data.cmvn import CMVN
from src.data.feature_extraction import FeatureExtractor

class TestFeatureScalingCMVN:
    """Test feature scaling with CMVN"""

    def test_cmvn_normalization_logic(self):
        """Test the mathematical correctness of CMVN normalization"""
        feature_dim = 64
        num_utterances = 10
        time_steps = 100
        
        # Create features with high mean and variance
        features_list = [torch.randn(feature_dim, time_steps) * 5.0 + 10.0 for _ in range(num_utterances)]
        
        cmvn = CMVN()
        cmvn.compute_stats(features_list)
        
        # Test normalization
        test_feat = torch.randn(feature_dim, time_steps) * 5.0 + 10.0
        normalized = cmvn.normalize(test_feat)
        
        # Normalized features should have approximately 0 mean and 1 std
        # (calculated over the same distribution)
        assert torch.abs(normalized.mean()) < 0.1
        assert torch.abs(normalized.std() - 1.0) < 0.1

    def test_feature_extractor_with_cmvn(self):
        """Test that FeatureExtractor correctly applies CMVN"""
        feature_dim = 40
        cmvn = CMVN()
        # Mock some stats
        cmvn.mean = torch.full((feature_dim,), 10.0)
        cmvn.std = torch.full((feature_dim,), 2.0)
        cmvn.count = torch.tensor(100)
        cmvn._initialized = True
        
        extractor = FeatureExtractor(
            n_mels=feature_dim,
            cmvn=cmvn
        )
        
        # Dummy waveform
        waveform = torch.randn(16000) # 1s
        features = extractor(waveform) # (1, feature_dim, time)
        
        # Features before CMVN (mel-spectrogram in dB) are usually > 0
        # With mean=10 subtracted, they should be shifted
        # We can verify by comparing with manual normalization
        
        # Disable CMVN to get raw features
        extractor.cmvn = None
        raw_features = extractor(waveform)
        
        # Manual normalize
        expected = (raw_features - 10.0) / 2.0
        
        # Re-enable and test
        extractor.cmvn = cmvn
        norm_features = extractor(waveform)
        
        assert torch.allclose(norm_features, expected, atol=1e-5)

    def test_cmvn_4d_input(self):
        """Test CMVN handles (B, 1, C, T) input shapes"""
        C, T = 40, 100
        B = 8
        cmvn = CMVN()
        cmvn.mean = torch.zeros(C)
        cmvn.std = torch.ones(C)
        cmvn._initialized = True
        
        input_4d = torch.randn(B, 1, C, T)
        normalized = cmvn.normalize(input_4d)
        
        assert normalized.shape == (B, 1, C, T)
        assert torch.allclose(normalized, input_4d) # since mean=0, std=1
