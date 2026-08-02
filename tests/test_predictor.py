import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from predictor import OutputLengthPredictor, PromptFeatureExtractor
import numpy as np


def test_feature_extraction():
    """Test prompt feature extraction."""
    extractor = PromptFeatureExtractor()

    prompt = "What is machine learning?"
    features = extractor.extract_features(prompt)

    # Should return 8 features
    assert features.shape == (1, 8)
    assert np.all(features >= 0)
    assert np.all(features <= 1.0)


def test_predictor_training():
    """Test predictor training."""
    predictor = OutputLengthPredictor()

    prompts = [
        "What is ML?",
        "Explain neural networks in detail",
        "Write a function",
    ]
    lengths = [50, 500, 150]

    predictor.train(prompts, lengths)
    assert predictor.is_trained


def test_predictor_prediction():
    """Test prediction functionality."""
    predictor = OutputLengthPredictor()

    prompts = [
        "What is ML?",
        "Explain neural networks in detail",
        "Write a function",
    ] * 10
    lengths = [50, 500, 150] * 10

    predictor.train(prompts, lengths)

    # Predict on similar prompts
    pred_short = predictor.predict("What is deep learning?")
    pred_long = predictor.predict("Give a comprehensive explanation of transformer architectures")

    # Longer prompts should predict longer outputs
    assert pred_long > pred_short

    # All predictions should be positive
    assert pred_short > 0
    assert pred_long > 0


def test_batch_prediction():
    """Test batch prediction."""
    predictor = OutputLengthPredictor()

    prompts = [
        "What is ML?",
        "Explain neural networks",
        "Write a function",
    ] * 5
    lengths = [50, 500, 150] * 5

    predictor.train(prompts, lengths)

    test_prompts = [
        "Short question?",
        "Very long and detailed explanation requested",
        "Medium prompt",
    ]

    predictions = predictor.predict_batch(test_prompts)
    assert len(predictions) == 3
    assert np.all(predictions > 0)


if __name__ == "__main__":
    test_feature_extraction()
    print("✓ Feature extraction test passed")

    test_predictor_training()
    print("✓ Predictor training test passed")

    test_predictor_prediction()
    print("✓ Predictor prediction test passed")

    test_batch_prediction()
    print("✓ Batch prediction test passed")

    print("\nAll predictor tests passed!")
