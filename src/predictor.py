import numpy as np
from sklearn.linear_model import Ridge
import re
from collections import Counter


class PromptFeatureExtractor:
    """Extract features from prompts to predict output length."""

    def extract_features(self, prompt: str) -> np.ndarray:
        """Extract numeric features from prompt text."""
        features = []

        # Feature 1: Prompt length (normalized)
        prompt_len = len(prompt)
        features.append(prompt_len / 10000.0)

        # Feature 2: Word count
        word_count = len(prompt.split())
        features.append(word_count / 2000.0)

        # Feature 3: Vocabulary entropy (unique words / total words)
        words = prompt.lower().split()
        unique_words = len(set(words))
        vocab_entropy = unique_words / max(len(words), 1)
        features.append(vocab_entropy)

        # Feature 4: Presence of question marks (binary indicator)
        question_count = prompt.count("?")
        features.append(min(question_count / 5.0, 1.0))

        # Feature 5: Presence of code blocks
        code_markers = len(re.findall(r"```|{|}", prompt))
        features.append(min(code_markers / 20.0, 1.0))

        # Feature 6: Presence of instructions (imperative verbs)
        imperative_verbs = [
            "generate",
            "write",
            "create",
            "explain",
            "summarize",
            "translate",
            "convert",
        ]
        verb_count = sum(
            1 for verb in imperative_verbs if verb in prompt.lower()
        )
        features.append(min(verb_count / 5.0, 1.0))

        # Feature 7: Punctuation density
        punctuation_count = sum(1 for c in prompt if c in ".!?,;:")
        punct_density = punctuation_count / max(len(prompt), 1)
        features.append(punct_density)

        # Feature 8: Uppercase density (indicates acronyms/emphasis)
        uppercase_count = sum(1 for c in prompt if c.isupper())
        uppercase_density = uppercase_count / max(len(prompt), 1)
        features.append(uppercase_density)

        return np.array(features, dtype=np.float32).reshape(1, -1)


class OutputLengthPredictor:
    """Predict output token length using Ridge regression."""

    def __init__(self, alpha=1.0):
        self.model = Ridge(alpha=alpha)
        self.feature_extractor = PromptFeatureExtractor()
        self.is_trained = False

    def train(self, prompts: list[str], output_lengths: list[int]):
        """Train the predictor on prompt-output pairs."""
        X = np.vstack([self.feature_extractor.extract_features(p) for p in prompts])
        y = np.array(output_lengths, dtype=np.float32)
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, prompt: str) -> float:
        """Predict output length for a prompt."""
        if not self.is_trained:
            # Default prediction if not trained
            return 100.0

        X = self.feature_extractor.extract_features(prompt)
        predicted_length = self.model.predict(X)[0]
        # Ensure positive prediction
        return max(predicted_length, 1.0)

    def predict_batch(self, prompts: list[str]) -> np.ndarray:
        """Predict output lengths for multiple prompts."""
        if not self.is_trained:
            return np.array([100.0] * len(prompts))

        X = np.vstack([self.feature_extractor.extract_features(p) for p in prompts])
        predictions = self.model.predict(X)
        return np.maximum(predictions, 1.0)
