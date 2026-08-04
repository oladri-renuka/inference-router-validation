import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error
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

        # Feature 4: Question density
        question_count = prompt.count("?")
        features.append(min(question_count / 5.0, 1.0))

        # Feature 5: Code block presence
        code_markers = len(re.findall(r"```|{|}", prompt))
        features.append(min(code_markers / 20.0, 1.0))

        # Feature 6: Imperative verbs (task-oriented)
        imperative_verbs = [
            "generate", "write", "create", "explain", "summarize",
            "translate", "convert", "list", "describe", "analyze"
        ]
        verb_count = sum(1 for verb in imperative_verbs if verb in prompt.lower())
        features.append(min(verb_count / 5.0, 1.0))

        # Feature 7: Punctuation density
        punctuation_count = sum(1 for c in prompt if c in ".!?,;:")
        punct_density = punctuation_count / max(len(prompt), 1)
        features.append(punct_density)

        # Feature 8: Uppercase density
        uppercase_count = sum(1 for c in prompt if c.isupper())
        uppercase_density = uppercase_count / max(len(prompt), 1)
        features.append(uppercase_density)

        # Feature 9: Sentence count (period-based)
        sentence_count = prompt.count(".") + prompt.count("?") + prompt.count("!")
        features.append(min(sentence_count / 10.0, 1.0))

        # Feature 10: Number presence (digits indicate data-heavy requests)
        digit_count = sum(1 for c in prompt if c.isdigit())
        features.append(min(digit_count / 20.0, 1.0))

        return np.array(features, dtype=np.float32).reshape(1, -1)


class OutputLengthPredictor:
    """Predict output token length using Ridge regression."""

    def __init__(self, alpha=1.0, model_type="ridge"):
        self.alpha = alpha
        self.model_type = model_type

        if model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            self.model = Lasso(alpha=alpha/10)
        elif model_type == "elasticnet":
            self.model = ElasticNet(alpha=alpha/10, l1_ratio=0.5)
        else:
            self.model = Ridge(alpha=alpha)

        self.feature_extractor = PromptFeatureExtractor()
        self.is_trained = False
        self.X_train = None
        self.y_train = None

    def train(self, prompts: list[str], output_lengths: list[int]):
        """Train the predictor on prompt-output pairs."""
        X = np.vstack([self.feature_extractor.extract_features(p) for p in prompts])
        y = np.array(output_lengths, dtype=np.float32)
        self.model.fit(X, y)
        self.X_train = X
        self.y_train = y
        self.is_trained = True

    def predict(self, prompt: str) -> float:
        """Predict output length for a prompt."""
        if not self.is_trained:
            return 100.0

        X = self.feature_extractor.extract_features(prompt)
        predicted_length = self.model.predict(X)[0]
        return max(predicted_length, 1.0)

    def predict_batch(self, prompts: list[str]) -> np.ndarray:
        """Predict output lengths for multiple prompts."""
        if not self.is_trained:
            return np.array([100.0] * len(prompts))

        X = np.vstack([self.feature_extractor.extract_features(p) for p in prompts])
        predictions = self.model.predict(X)
        return np.maximum(predictions, 1.0)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        if not self.is_trained:
            return 0.0
        predictions = self.model.predict(X)
        return r2_score(y, predictions)

    def mae(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute mean absolute error."""
        if not self.is_trained:
            return 0.0
        predictions = self.model.predict(X)
        return mean_absolute_error(y, predictions)

    def confusion_matrix_at_threshold(self, X: np.ndarray, y: np.ndarray, threshold: float = 500.0) -> dict:
        """
        Compute confusion matrix for binary classification at threshold.

        Returns:
            {
                "true_positive": count of y >= threshold, pred >= threshold,
                "true_negative": count of y < threshold, pred < threshold,
                "false_positive": count of y < threshold, pred >= threshold,
                "false_negative": count of y >= threshold, pred < threshold,
                "accuracy": (TP + TN) / total,
                "precision": TP / (TP + FP),
                "recall": TP / (TP + FN)
            }
        """
        predictions = self.model.predict(X)

        actual_long = y >= threshold
        pred_long = predictions >= threshold

        tp = np.sum(actual_long & pred_long)
        tn = np.sum(~actual_long & ~pred_long)
        fp = np.sum(~actual_long & pred_long)
        fn = np.sum(actual_long & ~pred_long)

        total = len(y)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "true_positive": int(tp),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
