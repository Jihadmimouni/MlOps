# DevOps & MLOps - CI/CD Implementation Report

## Project Overview
This report documents the complete implementation of CI/CD pipeline for the Iris Classifier ML application, covering testing, linting, containerization, and automated workflows.

**Repository**: [Your GitHub Repository Link]  
**Source Repository**: [Original Repository Link]  
**Date**: November 17, 2025

---

## Task 1: Prepare the ML Project

### Actions Taken
1. **Forked/Cloned the repository** from the source
2. **Inspected repository structure** to understand the project layout
3. **Verified requirements.txt** exists with all necessary dependencies

### Repository Structure
```
ml-app/
├── .github/
│   └── workflows/
│       └── ci.yml
├── models/
│   └── iris_classifier.pkl
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_model.py
├── .dockerignore
├── .flake8
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── REPORT.md
```

### Requirements Verification
The `requirements.txt` file contains all necessary dependencies:
- scikit-learn==1.3.0
- pandas==2.0.3
- numpy==1.24.3
- matplotlib==3.7.1
- seaborn==0.12.2
- joblib==1.2.0
- pytest==7.3.1
- pytest-cov==4.1.0
- black==23.3.0
- flake8==6.0.0

![fork](./screenshots/Pasted%20image.png)
![clone](./screenshots/Pasted%20image%20(2).png)
![requirments](./screenshots/Pasted%20image%20(3).png)
---

## Task 2: Run the App Locally

### Virtual Environment Setup

Created and activated a Python virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate
```

### Dependencies Installation

```bash
# Upgrade pip
python -m pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt
```

![venv](./screenshots/Pasted%20image%20(4).png)
![install dependency](./screenshots/Pasted%20image%20(5).png)

### Running the Application

#### Training the Model

```bash
# Run training script
python src/train.py
```

**Training Output**:
- Successfully loaded Iris dataset (150 samples, 4 features)
- Training set: 120 samples
- Test set: 30 samples
- Model trained using Logistic Regression
- **Final Accuracy: 96.67%**
- Model saved to: `models/iris_classifier.pkl`
- Generated plots: `confusion_matrix.png`, `feature_importance.png`

![Training Output](./screenshots/Pasted%20image%20(6).png)

#### Making Predictions

```bash
# Run prediction script
python src/predict.py
```

**Prediction Output**:
The application successfully loaded the trained model and made predictions on example data with confidence scores.

![Prediction Output](./screenshots/Pasted%20image%20(7).png)

### How to Test Locally

1. **Ensure virtual environment is activated**
2. **Run training**: `python src/train.py`
3. **Check outputs**:
   - Model file: `models/iris_classifier.pkl`
   - Confusion matrix: `confusion_matrix.png`
   - Feature importance: `feature_importance.png`
4. **Run predictions**: `python src/predict.py`
5. **Verify predictions** are displayed with confidence scores

---

## Task 3: Write Unit Tests

### Implementation

Created a comprehensive test suite in the `tests/` directory using pytest framework.

**Test File**: `tests/test_model.py`  
**Framework**: pytest 7.3.1  
**Total Tests**: 3 unit tests

### Test Configuration Files

1. **pytest.ini** - Test configuration:
   - Test discovery patterns (`test_*.py`)
   - Python path configuration (`pythonpath = .`)
   - Verbose output settings
   - Test paths specification

2. **Package Structure**:
   - Added `src/__init__.py` - Makes src a proper Python package
   - Added `tests/__init__.py` - Makes tests a proper Python package

### Test Cases Implemented

#### Test 1: `test_data_format_and_structure`
**Purpose**: Data format validation and sanity checks

**Validations**:
- Data types are numpy arrays
- Correct feature dimensions (4 features for Iris dataset)
- Total samples match expected count (150 samples)
- Class labels are valid (0, 1, 2 for 3 Iris species)
- No missing values in data

```python
def test_data_format_and_structure():
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)
    
    # Check data types
    assert isinstance(X_train, np.ndarray), "X_train should be numpy array"
    assert isinstance(y_train, np.ndarray), "y_train should be numpy array"
    
    # Check dimensions
    assert X_train.shape[1] == 4, "Should have 4 features"
    assert len(X_train) + len(X_test) == 150, "Total samples should be 150"
    
    # Check class labels
    unique_classes = np.unique(y_train)
    assert len(unique_classes) == 3, "Should have 3 classes"
    assert set(unique_classes) == {0, 1, 2}, "Classes should be 0, 1, 2"
    
    # Verify no missing values
    assert not np.isnan(X_train).any(), "Training data should not contain NaN"
```

#### Test 2: `test_model_training_sanity`
**Purpose**: Model training validation and performance check

**Validations**:
- Model initializes in untrained state
- Model successfully trains on data
- Training state flag updates correctly
- Achieves reasonable accuracy (>85% for Iris dataset)
- Classification report contains expected metrics

```python
def test_model_training_sanity():
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)
    classifier = IrisClassifier(random_state=42)
    
    # Check initial state
    assert not classifier.is_trained, "Model should not be trained initially"
    
    # Train the model
    classifier.train(X_train, y_train)
    assert classifier.is_trained, "Model should be marked as trained"
    
    # Evaluate performance
    accuracy, report = classifier.evaluate(X_test, y_test)
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert accuracy > 0.85, f"Model accuracy {accuracy:.2f} is too low"
```

#### Test 3: `test_prediction_output_validation`
**Purpose**: Prediction functionality and error handling

**Validations**:
- Predictions return numpy array
- Prediction count matches input samples
- All predictions are valid class labels (0, 1, or 2)
- Predictions are integer type
- Error handling: ValueError raised when predicting before training

```python
def test_prediction_output_validation():
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=42)
    classifier = IrisClassifier(random_state=42)
    classifier.train(X_train, y_train)
    
    # Make predictions
    test_samples = X_test[:10]
    predictions = classifier.predict(test_samples)
    
    # Validate output
    assert isinstance(predictions, np.ndarray), "Predictions should be array"
    assert len(predictions) == 10, "Should return predictions for 10 samples"
    assert all(pred in [0, 1, 2] for pred in predictions), "Valid class labels"
    
    # Test error handling
    untrained_classifier = IrisClassifier()
    with pytest.raises(ValueError, match="Model must be trained"):
        untrained_classifier.predict(test_samples)
```

### Running Tests

```bash
# Run all tests with verbose output
pytest --verbose

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Results

All 3 tests pass successfully:

```
======================================================= test session starts ========================================================
platform linux -- Python 3.11.14, pytest-7.3.1
rootdir: /home/gl1tch/gitClones/DevOps-MLOps-Labs/session2/ml-app
configfile: pytest.ini
testpaths: tests
collected 3 items

tests/test_model.py::test_data_format_and_structure PASSED                                                                   [ 33%]
tests/test_model.py::test_model_training_sanity PASSED                                                                       [ 66%]
tests/test_model.py::test_prediction_output_validation PASSED                                                                [100%]

======================================================== 3 passed in 0.87s =========================================================
```

![Test Results Screenshot](./screenshots/Pasted%20image%20(8).png)

### Test Coverage

Tests cover the following modules:
- `src/data_loader.py` - Data loading and preprocessing
- `src/model.py` - Model training, prediction, and evaluation
- Integration between data loading and model training

---

## Task 4: Linting & Formatting

### Linter Setup

Added **flake8** for automated code style checking and PEP 8 compliance.

### Configuration File: `.flake8`

Created comprehensive flake8 configuration:

```ini
[flake8]
# Maximum line length
max-line-length = 100

# Exclude patterns
exclude =
    .git,
    __pycache__,
    .venv,
    venv,
    env,
    *.egg-info,
    build,
    dist,
    .pytest_cache

# Error codes to ignore (black compatibility)
ignore = E203,W503

# Maximum complexity
max-complexity = 10

# Show source code for each error
show-source = True

# Show the relevant error codes
show-pep8 = True
```

### Style Violations Fixed

Fixed all flake8 violations across the codebase:

1. **src/data_loader.py**:
   - Removed unused `Optional` import
   - Added 2 blank lines between functions (PEP 8)
   - Fixed line length (split function signature)
   - Removed f-string without placeholders

2. **src/model.py**:
   - Added 2 blank lines before class definition

3. **src/utils.py**:
   - Added 2 blank lines between functions

4. **src/train.py** & **src/predict.py**:
   - Removed unused imports (`sys`, `os`, `numpy`)
   - Added 2 blank lines after function definitions

5. **tests/test_model.py**:
   - Fixed import order (module-level imports at top)
   - Removed whitespace on blank lines
   - Cleaned up sys.path hacks

### Running the Linter

```bash
# Check all source and test files
flake8 src/ tests/

# Check specific file
flake8 src/model.py

# Show statistics
flake8 --statistics src/ tests/
```

### Linting Results

All code now passes flake8 checks with **zero violations**:

```bash
$ flake8 src/ tests/
# No output = all checks passed ✓
```

![Flake8 before](./screenshots/Pasted%20image%20(9).png)

![Flake8 after](./screenshots/Pasted%20image%20(11).png)


### Benefits of Linting

- **Code Consistency**: Enforces uniform coding style across the project
- **Early Bug Detection**: Catches potential issues before runtime
- **PEP 8 Compliance**: Follows Python's official style guide
- **Maintainability**: Makes code easier to read and maintain
- **Team Collaboration**: Reduces style debates and merge conflicts

---

## Task 5: GitHub Actions CI Workflow

### Workflow File

Created comprehensive CI/CD pipeline in `.github/workflows/ci.yml`.

### Workflow Configuration

**Name**: CI Pipeline  
**Triggers**: 
- Push to `main` and `develop` branches
- Pull requests to `main` and `develop` branches

**Runner**: ubuntu-latest  
**Python Version**: 3.11

### Pipeline Steps

#### Step 1: Checkout Code
```yaml
- name: Checkout code
  uses: actions/checkout@v4
```
- Uses official GitHub action to checkout repository
- Fetches complete git history

#### Step 2: Set Up Python
```yaml
- name: Set up Python 3.11
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'
    cache: 'pip'
```
- Uses official `setup-python` action (v5)
- Installs Python 3.11
- Enables pip caching for faster builds

#### Step 3: Install Dependencies
```yaml
- name: Install dependencies
  run: |
    python -m pip install --upgrade pip
    pip install -r requirements.txt
```
- Upgrades pip to latest version
- Installs all project dependencies

#### Step 4: Run Linter
```yaml
- name: Run flake8 linter
  run: |
    flake8 src/ tests/
  continue-on-error: false
```
- Executes flake8 on all source and test files
- Fails the build if style violations found

#### Step 5: Run Tests with Coverage
```yaml
- name: Run tests with pytest
  run: |
    pytest --verbose --junit-xml=test-results.xml \
           --cov=src --cov-report=xml --cov-report=html
  continue-on-error: false
```
- Runs all tests with verbose output
- Generates JUnit XML test results
- Creates XML and HTML coverage reports
- Coverage includes all `src/` modules

#### Step 6: Upload Test Results
```yaml
- name: Upload test results
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: test-results
    path: test-results.xml
    retention-days: 30
```
- Uses `upload-artifact@v4` action
- Uploads JUnit XML test results
- Always runs (even if tests fail)
- Retained for 30 days

#### Step 7: Upload Coverage Report
```yaml
- name: Upload coverage report
  uses: actions/upload-artifact@v4
  if: always()
  with:
    name: coverage-report
    path: |
      coverage.xml
      htmlcov/
    retention-days: 30
```
- Uploads both XML and HTML coverage reports
- Always runs (even if tests fail)
- Retained for 30 days

#### Step 8: Build Docker Image
```yaml
- name: Build Docker image
  run: |
    docker build -t iris-classifier:${{ github.sha }} .
    docker save iris-classifier:${{ github.sha }} -o iris-classifier-image.tar
```
- Builds Docker image with commit SHA tag
- Saves image as tar archive for artifact upload
- Image tagged as `iris-classifier:<commit-sha>`

#### Step 9: Upload Docker Image
```yaml
- name: Upload Docker image artifact
  uses: actions/upload-artifact@v4
  with:
    name: docker-image
    path: iris-classifier-image.tar
    retention-days: 7
```
- Uploads Docker image as tar file
- Makes image downloadable from workflow run
- Retained for 7 days

#### Step 10: Display Summary
```yaml
- name: Display summary
  if: always()
  run: |
    echo "### CI Pipeline Summary" >> $GITHUB_STEP_SUMMARY
    echo "- **Python Version:** 3.11" >> $GITHUB_STEP_SUMMARY
    echo "- **Docker Image:** iris-classifier:${{ github.sha }}" >> $GITHUB_STEP_SUMMARY
```
- Creates workflow summary visible in GitHub UI
- Shows key information about the build

### Artifacts Generated

The CI workflow produces three types of artifacts:

1. **test-results** (30 days retention)
   - JUnit XML format test results
   - Can be integrated with test reporting tools
   - Shows test execution details

2. **coverage-report** (30 days retention)
   - XML coverage report (`coverage.xml`)
   - HTML coverage report (`htmlcov/`)
   - Shows code coverage metrics and uncovered lines

3. **docker-image** (7 days retention)
   - Docker image as tar archive
   - Tagged with commit SHA
   - Ready for deployment or testing
   - Can be downloaded and loaded with `docker load`

### Running Workflow Locally

To test workflow steps before pushing:

```bash
# Step 1: Linting
flake8 src/ tests/

# Step 2: Testing with coverage
pytest --verbose --junit-xml=test-results.xml \
       --cov=src --cov-report=xml --cov-report=html

# Step 3: Build Docker image
docker build -t iris-classifier:local .

# Step 4: Save Docker image
docker save iris-classifier:local -o iris-classifier-image.tar
```

### CI Workflow Behavior

**On Push**:
- Workflow triggers automatically
- All steps execute in sequence
- Build fails if any step fails
- Artifacts uploaded regardless of test results
- Docker image only uploaded if build succeeds

**On Pull Request**:
- Workflow triggers for each commit
- Same steps as push trigger
- Results visible in PR checks
- Reviewers can download artifacts

**Branch Protection** (Recommended):
- Require CI to pass before merging
- Require code review approval
- Prevent direct pushes to main

![GitHub Actions Workflow](./screenshots/Pasted%20image%20(16).png)

![CI Pipeline Running](./screenshots/Pasted%20image%20(14).png)

![CI Pipeline Success](./screenshots/Pasted%20image%20(17).png)

![Artifacts Downloaded](./screenshots/Pasted%20image%20(18).png)

### Additional CI Improvements

Added to support CI/CD:
- `pytest-cov==4.1.0` for code coverage reporting
- `src/__init__.py` and `tests/__init__.py` for proper Python package structure
- Updated `pytest.ini` with `pythonpath = .` for correct module resolution
- Removed sys.path hacks from test files
- Optimized Docker layer caching

---

## Task 6: Containerize the App

### Dockerfile Implementation

Enhanced the Dockerfile with production-ready features and best practices.

### Dockerfile Configuration

**Base Image**: `python:3.11-slim`  
**Working Directory**: `/app`  
**Exposed Port**: 8000 (for future API)  
**Default Command**: `python src/train.py`

### Complete Dockerfile

```dockerfile
# Use official Python runtime as base image
FROM python:3.11-slim

# Set maintainer label
LABEL maintainer="ml-app"
LABEL description="Iris Classifier ML Application"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code to container
COPY src/ ./src/
COPY pytest.ini .
COPY .flake8 .

# Create directories for models and outputs
RUN mkdir -p models

# Expose port (for future prediction API)
EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command: Run training
CMD ["python", "src/train.py"]
```

### Dockerfile Features

1. **Environment Variables**:
   - `PYTHONUNBUFFERED=1` - Ensures real-time log output
   - `PYTHONDONTWRITEBYTECODE=1` - Prevents `.pyc` file creation
   - `PIP_NO_CACHE_DIR=1` - Reduces image size
   - `PIP_DISABLE_PIP_VERSION_CHECK=1` - Faster pip operations

2. **Layer Optimization**:
   - Requirements installed first for better caching
   - Only necessary files copied (via `.dockerignore`)
   - Multi-stage build ready for future optimizations

3. **Health Check**:
   - Configured for container health monitoring
   - Runs every 30 seconds
   - Useful for orchestration tools (Kubernetes, Docker Swarm)

4. **Labels**:
   - Maintainer and description metadata
   - Helps with image organization and discovery

5. **Port Exposure**:
   - Port 8000 exposed for future Flask/FastAPI integration
   - Ready for prediction API deployment

### Docker Ignore File

The `.dockerignore` file excludes unnecessary files:

```
# Python cache
__pycache__/
*.py[cod]
*$py.class

# Virtual environments
.venv/
venv/
env/

# Testing
.pytest_cache/
.coverage
htmlcov/
test-results.xml
coverage.xml

# IDE
.vscode/
.idea/

# Git
.git/
.gitignore
.github/

# Documentation
README.md
REPORT.md

# Generated files
confusion_matrix.png
feature_importance.png
*.pkl

# Docker
Dockerfile
.dockerignore
```

**Benefits**:
- Reduces image size significantly
- Speeds up build process
- Prevents sensitive data inclusion
- Excludes development-only files

### Building the Docker Image

```bash
# Build with single tag
docker build -t iris-classifier:latest .

# Build with multiple tags
docker build -t iris-classifier:latest -t iris-classifier:v1.0 .

# Build with custom name
docker build -t myrepo/iris-classifier:latest .
```

**Build Output**:
```
[+] Building 234.8s (13/13) FINISHED
 => [internal] load build definition from Dockerfile                    0.0s
 => [internal] load metadata for docker.io/library/python:3.11-slim     3.1s
 => [1/8] FROM docker.io/library/python:3.11-slim                      31.2s
 => [2/8] WORKDIR /app                                                  0.4s
 => [3/8] COPY requirements.txt .                                       0.0s
 => [4/8] RUN pip install --upgrade pip && pip install -r requirements 195.9s
 => [5/8] COPY src/ ./src/                                              0.1s
 => [6/8] COPY pytest.ini .                                             0.1s
 => [7/8] COPY .flake8 .                                                0.0s
 => [8/8] RUN mkdir -p models                                           0.2s
 => exporting to image                                                  3.7s
 => => naming to docker.io/library/iris-classifier:latest              0.0s
```

**Image Details**:
- Image Size: 593MB
- Build Time: ~235 seconds (first build)
- Base Image: python:3.11-slim (lightweight)

![Docker Build Process](./screenshots/Pasted%20image%20(10).png)

### Verifying the Image

```bash
# List Docker images
docker images | grep iris-classifier

# Output:
# iris-classifier    latest    9160d0223d39   2 minutes ago   593MB
# iris-classifier    v1.0      9160d0223d39   2 minutes ago   593MB
```

![Docker Images List](./screenshots/Pasted%20image%20(12).png)

### Running the Dockerized Application

#### Training Mode (Default)

```bash
# Run training with volume mounts to save outputs
docker run --name iris-training --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd):/app/output \
  iris-classifier:latest
```

**Training Output**:
```
Starting Iris Classifier Training...
Loading Iris dataset...
Successfully loaded Iris dataset
   Features: 4, Samples: 150
   Training set: 120 samples
   Test set: 30 samples
Training Logistic Regression model...
Evaluating model...
Model Accuracy: 0.9667

Classification Report:
              precision    recall  f1-score   support
           0       1.00      1.00      1.00        10
           1       1.00      0.90      0.95        10
           2       0.91      1.00      0.95        10
    accuracy                           0.97        30

Saving model...
Generating evaluation plots...
Training completed successfully!
Model saved to: models/iris_classifier.pkl
Plots saved: confusion_matrix.png, feature_importance.png
```

![Docker Training Output](./screenshots/Pasted%20image%20(13).png)


#### Prediction Mode

```bash
# Run predictions using trained model
docker run --name iris-predict --rm \
  -v $(pwd)/models:/app/models \
  iris-classifier:latest python src/predict.py
```

**Prediction Output**:
```
Iris Classifier Prediction
Model loaded successfully!

 Example Predictions:
Features: [sepal length, sepal width, petal length, petal width]

Example 1: [5.1, 3.5, 1.4, 0.2]
Prediction: setosa
Probabilities:
  setosa: 0.9784
  versicolor: 0.0216
  virginica: 0.0000

Example 2: [6.7, 3.0, 5.2, 2.3]
Prediction: virginica
Probabilities:
  setosa: 0.0001
  versicolor: 0.0924
  virginica: 0.9075

Example 3: [5.9, 3.0, 4.2, 1.5]
Prediction: versicolor
Probabilities:
  setosa: 0.0183
  versicolor: 0.8789
  virginica: 0.1028
```

![Docker Prediction Output](./screenshots/Pasted%20image%20(15).png)

### Docker Commands Reference

```bash
# List images
docker images | grep iris-classifier

# Run with custom command
docker run --rm iris-classifier:latest python src/predict.py

# Interactive shell
docker run -it --rm iris-classifier:latest /bin/bash

# View container logs
docker logs iris-training

# Stop running container
docker stop iris-training

# Remove container
docker rm iris-training

# Remove image
docker rmi iris-classifier:latest

# Inspect image details
docker inspect iris-classifier:latest

# Save image to file
docker save iris-classifier:latest -o iris-classifier-image.tar

# Load image from file
docker load -i iris-classifier-image.tar

# Push to registry (after login)
docker push myrepo/iris-classifier:latest
```

### Volume Mounts Explained

The containerized application uses volume mounts to persist data:

1. **Models Volume**: `-v $(pwd)/models:/app/models`
   - **Purpose**: Saves trained model files to host machine
   - **Benefit**: Model persists after container stops
   - **Use Case**: Reuse trained models without retraining

2. **Output Volume**: `-v $(pwd):/app/output`
   - **Purpose**: Saves plots and reports to host machine
   - **Benefit**: Results accessible outside container
   - **Use Case**: View visualizations and analysis results

### Container Benefits

1. **Reproducibility**: Exact same environment everywhere (dev, staging, prod)
2. **Isolation**: No dependency conflicts with host system
3. **Portability**: Run on any system with Docker (Linux, Mac, Windows)
4. **CI/CD Integration**: Easy to integrate in automated pipelines
5. **Scalability**: Can run multiple containers simultaneously
6. **Version Control**: Tag images for different application versions
7. **Resource Management**: Control CPU and memory allocation
8. **Easy Deployment**: Deploy to Kubernetes, Docker Swarm, cloud platforms

---

## Summary & Conclusions

### Deliverables Completed

✅ **Task 1**: Repository forked/created with proper structure  
✅ **Task 2**: Application runs locally with virtual environment  
✅ **Task 3**: 3 comprehensive unit tests with pytest  
✅ **Task 4**: Flake8 linter configured, all code passes  
✅ **Task 5**: GitHub Actions CI workflow with all required steps  
✅ **Task 6**: Dockerfile created, image built and tested  
