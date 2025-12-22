import pytest
from src.evaluation.streaming_detector import StreamingDetector

def test_n_of_m_smoothing_activation():
    """Test that detector activates ONLY when N of M frames exceed threshold."""
    # N=3, M=5
    detector = StreamingDetector(
        threshold_on=0.5,
        vote_window=5,
        vote_threshold=3,
        lockout_ms=0
    )
    
    # 1. First high score: 1/5 -> No
    assert not detector.step(0.9, 100)
    
    # 2. Second high score: 2/5 -> No
    assert not detector.step(0.9, 200)
    
    # 3. Low score: 2/5 -> No
    assert not detector.step(0.1, 300)
    
    # 4. Third high score: 3/5 -> YES
    # Buffer: [0.9, 0.9, 0.1, 0.9] -> 3 votes >= 0.5
    assert detector.step(0.9, 400)
    assert detector.is_active

def test_n_of_m_smoothing_sliding():
    """Test that votes expire as window slides."""
    # N=2, M=3
    detector = StreamingDetector(
        threshold_on=0.5,
        vote_window=3,
        vote_threshold=2,
        lockout_ms=0
    )
    
    # T=1: High (1/3)
    assert not detector.step(0.9, 100)
    
    # T=2: Low (1/3)
    assert not detector.step(0.1, 200)
    
    # T=3: Low (1/3) -> Oldest high score is still in buffer? 
    # Buffer: [0.9, 0.1, 0.1] -> 1 vote.
    assert not detector.step(0.1, 300)
    
    # T=4: High. Buffer: [0.1, 0.1, 0.9] -> 1 vote (First high score dropped).
    assert not detector.step(0.9, 400)
    
    # T=5: High. Buffer: [0.1, 0.9, 0.9] -> 2 votes -> YES
    assert detector.step(0.9, 500)

def test_lockout_prevents_retrigger():
    """Test lockout logic."""
    detector = StreamingDetector(
        threshold_on=0.5,
        vote_window=1, # Immediate
        vote_threshold=1,
        lockout_ms=1000
    )
    
    # Trigger
    assert detector.step(0.9, 0)
    assert detector.is_active
    assert detector.locked_until_ms == 1000
    
    # Immediate next frame (high score) -> Should be ignored/False because already active/locked?
    # Logic: if in lockout, return False.
    assert not detector.step(0.9, 100)
    
    # After lockout
    # But detector stays active until votes_off condition met.
    # Hysteresis logic: 
    # if active: check votes_off.
    
    # Let's drop signal to deactivate
    detector.step(0.1, 1100) 
    # Buffer [0.1] -> votes_off = 1 >= 1 -> Deactivate
    assert not detector.is_active
    
    # Now retrigger
    assert detector.step(0.9, 1200)
