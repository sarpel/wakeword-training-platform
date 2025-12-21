# Product Guidelines - Wakeword Training Platform

## Voice and Tone
*   **Technical & Precise:** The primary mode of communication is clear, unambiguous engineering prose. Use standard machine learning and signal processing terminology correctly (e.g., use "Mel-Frequency Cepstral Coefficients" or "F1-Score" rather than simplified approximations).
*   **Accuracy-First:** Documentation and UI labels should prioritize technical correctness over marketing fluff. If a setting is "Experimental," it must be clearly labeled as such with a brief explanation of the risks.

## Visual Identity
*   **The "Mission Control" Aesthetic:** The UI (Gradio) should feel like a high-tech dashboard. Prefer dark themes with high-contrast neon accents (e.g., cyan/amber) for critical indicators.
*   **Data-Dense Visualization:** Prioritize information density. Use interactive graphs (Plotly), real-time scrolling logs, and multi-column layouts to provide a comprehensive view of the training state at a glance.

## Code and Engineering Standards
*   **Robust & Safe:** All new Python code MUST use Type Hints. Error handling should be explicit; avoid "catch-all" except blocks. Configurable values must be validated via Pydantic or similar mechanisms.
*   **Highly Modular Architecture:** Adhere to a strict separation of concerns. The Training logic must be decoupled from the UI, and Data Preprocessing must be independent of the Model Architecture. This ensures that adding a new model type (e.g., a Transformer-based wakeword) requires minimal changes to the core pipeline.
*   **Performance-Critical Logic:** While maintaining Pythonic readability, performance bottlenecks (audio resampling, feature extraction, batch loading) must be optimized. Use vectorized NumPy/PyTorch operations and shared memory buffers where appropriate to minimize latency.
