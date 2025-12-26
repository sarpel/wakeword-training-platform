import numpy as np
import pytest

from src.evaluation.types import InferenceEngine, StageBase


def test_inference_engine_is_abstract():
    """Ensure InferenceEngine cannot be instantiated directly."""
    with pytest.raises(TypeError):
        InferenceEngine()


def test_stage_base_is_abstract():
    """Ensure StageBase cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StageBase()


class MockStage(StageBase):
    def predict(self, audio: np.ndarray) -> dict:
        return {"confidence": 0.9}

    @property
    def name(self) -> str:
        return "mock_stage"


def test_stage_implementation():
    """Test that a valid stage implementation works."""
    stage = MockStage()
    assert stage.name == "mock_stage"
    result = stage.predict(np.array([0.1, 0.2]))
    assert result["confidence"] == 0.9


class MockEngine(InferenceEngine):
    def __init__(self):
        self.stages = []

    def add_stage(self, stage: StageBase):
        self.stages.append(stage)

    def run(self, audio: np.ndarray) -> list:
        results = []
        for stage in self.stages:
            results.append(stage.predict(audio))
        return results


def test_engine_implementation():
    """Test that a valid engine implementation works."""
    engine = MockEngine()
    stage = MockStage()
    engine.add_stage(stage)
    results = engine.run(np.array([0.1, 0.2]))
    assert len(results) == 1
    assert results[0]["confidence"] == 0.9


from src.evaluation.streaming_detector import CascadeInferenceEngine


def test_cascade_inference_engine_real():
    """Test the actual CascadeInferenceEngine implementation."""
    engine = CascadeInferenceEngine()
    stage = MockStage()
    engine.add_stage(stage)

    audio = np.random.randn(16000)  # 1 second
    results = engine.run(audio)

    assert len(results) > 0
    assert results[0]["stage"] == "mock_stage"
    assert "confidence" in results[0]["result"]


from src.evaluation.stages import SentryInferenceStage
from src.models.architectures import MobileNetV3Wakeword


def test_sentry_inference_stage():
    """Test SentryInferenceStage with a real MobileNetV3 model."""
    model = MobileNetV3Wakeword(num_classes=2)
    model.eval()

    stage = SentryInferenceStage(model=model, name="sentry")
    assert stage.name == "sentry"

    audio = np.random.randn(16000)
    result = stage.predict(audio)

    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1.0


from src.evaluation.stages import JudgeInferenceStage
from src.models.huggingface import Wav2VecWakeword


def test_judge_inference_stage():
    """Test JudgeInferenceStage with a Wav2VecWakeword model."""
    # Using a small config to avoid heavy downloads if possible,
    # but here we'll just test the modularity
    model = Wav2VecWakeword(num_classes=2, pretrained=False)
    model.eval()

    stage = JudgeInferenceStage(model=model, name="judge")
    assert stage.name == "judge"

    audio = np.random.randn(16000)
    result = stage.predict(audio)

    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1.0


def test_cascade_handoff_logic():
    """Test that CascadeInferenceEngine correctly stops if a stage doesn't detect."""
    engine = CascadeInferenceEngine()

    class MockStageCustom(StageBase):
        def __init__(self, name, detect):
            self._name = name
            self.detect = detect

        def predict(self, audio):
            return {"detected": self.detect, "confidence": 1.0 if self.detect else 0.0}

        @property
        def name(self):
            return self._name

    stage1 = MockStageCustom("stage1", False)
    stage2 = MockStageCustom("stage2", True)

    engine.add_stage(stage1)
    engine.add_stage(stage2)

    audio = np.array([0.1])
    results = engine.run(audio)

    # Should stop after stage1 because it didn't detect
    assert len(results) == 1
    assert results[0]["stage"] == "stage1"


def test_full_sentry_judge_cascade():
    """Test a full Sentry -> Judge cascade."""
    s_model = MobileNetV3Wakeword(num_classes=2)
    j_model = Wav2VecWakeword(num_classes=2, pretrained=False)

    s_stage = SentryInferenceStage(s_model, threshold=0.0)  # Always detect for test
    j_stage = JudgeInferenceStage(j_model, threshold=0.0)  # Always detect for test

    engine = CascadeInferenceEngine()
    engine.add_stage(s_stage)
    engine.add_stage(j_stage)

    audio = np.random.randn(16000)
    results = engine.run(audio)

    assert len(results) == 2
    assert results[0]["stage"] == "sentry"
    assert results[1]["stage"] == "judge"
