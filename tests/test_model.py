"""
Unit tests for ML application core functionality.
Tests verify data format, model training, and prediction outputs.
"""

import pytest
import numpy as np

from src.data_loader import load_iris_data
from src.model import IrisClassifier


def test_data_format_and_structure():
    """
    Test 1: Data format validation
    Verifies that loaded data has correct shape, type, and class distribution.
    """
    X_train, X_test, y_train, y_test = load_iris_data(
        test_size=0.2, random_state=42
    )

    # Check data types
    assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be numpy array"

    # Check feature dimensions (Iris has 4 features)
    assert X_train.shape[1] == 4, "Should have 4 features"
    assert X_test.shape[1] == 4, "Test set should have 4 features"

    # Check total samples (Iris has 150 samples)
    assert len(X_train) + len(X_test) == 150, "Total samples should be 150"

    # Check class labels (Iris has 3 classes: 0, 1, 2)
    unique_classes = np.unique(y_train)
    assert len(unique_classes) == 3, "Should have 3 classes"
    assert set(unique_classes) == {0, 1, 2}, "Classes should be 0, 1, 2"

    # Verify no missing values
    assert not np.isnan(X_train).any(), "Training data should not contain NaN"
    assert not np.isnan(X_test).any(), "Test data should not contain NaN"


def test_model_training_sanity():
    """
    Test 2: Model training sanity check
    Verifies that model trains successfully and achieves reasonable accuracy.
    """
    X_train, X_test, y_train, y_test = load_iris_data(
        test_size=0.2, random_state=42
    )
    classifier = IrisClassifier(random_state=42)

    # Check initial state
    assert not classifier.is_trained, "Model should not be trained initially"

    # Train the model
    classifier.train(X_train, y_train)
    assert classifier.is_trained, "Model should be marked as trained"

    # Evaluate on test set
    accuracy, report = classifier.evaluate(X_test, y_test)

    # Sanity check: accuracy should be reasonable for Iris dataset
    # Iris is a simple dataset, so we expect > 85% accuracy
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    msg = f"Model accuracy {accuracy:.2f} is too low for Iris dataset"
    assert accuracy > 0.85, msg

    # Check that classification report is generated
    assert isinstance(report, str), "Classification report should be a string"
    assert "precision" in report.lower(), "Report should contain precision"
    assert "recall" in report.lower(), "Report should contain recall metric"


def test_prediction_output_validation():
    """
    Test 3: Prediction output validation
    Verifies that predictions have correct format and valid class labels.
    """
    X_train, X_test, y_train, y_test = load_iris_data(
        test_size=0.2, random_state=42
    )
    classifier = IrisClassifier(random_state=42)

    # Train model
    classifier.train(X_train, y_train)

    # Make predictions on a subset
    test_samples = X_test[:10]
    predictions = classifier.predict(test_samples)

    # Check prediction array properties
    assert isinstance(predictions, np.ndarray), "Predictions should be array"
    assert len(predictions) == 10, "Should return predictions for 10 samples"

    # Check prediction values are valid class labels
    assert all(pred in [0, 1, 2] for pred in predictions), \
        "All predictions should be valid class labels (0, 1, or 2)"

    # Check predictions are integers
    assert all(isinstance(pred, (np.integer, int)) for pred in predictions), \
        "Predictions should be integers"

    # Test error handling: predicting before training should raise error
    untrained_classifier = IrisClassifier()
    with pytest.raises(ValueError, match="Model must be trained"):
        untrained_classifier.predict(test_samples)
