# Voice Emotion Recognition ML Model

## About
End-to-end **speech emotion recognition** project in Python: audio preprocessing, feature extraction, model training, and evaluation.

The pipeline uses:
- **MFCC**, **ZCR**, and **RMS** audio features
- data augmentation (noise, pitch shift)
- a **1D CNN** model (TensorFlow / Keras) for emotion classification

## What this project shows
- Working with real audio datasets
- Feature engineering for speech signals
- Training and evaluating a deep learning classification model
- Saving and reusing trained model artifacts

## Tech stack
- Python
- TensorFlow / Keras
- librosa
- scikit-learn
- NumPy / pandas
- matplotlib / seaborn

## Project structure
- `prepareData.py` — dataset loading, augmentation, feature extraction
- `model.py` — CNN training, plots, confusion matrix, model export
- `load&Evaluate.py` — load saved model and run evaluation
- `Report ML Voice Emotion Recognition.docx` — project report

## Quick start
1) Install dependencies  
python -m venv .venv  
# Windows: .venv\Scripts\activate  
# Linux/Mac: source .venv/bin/activate  
python -m pip install --upgrade pip  
python -m pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow keras tqdm ipython

2) Prepare features  
python prepareData.py

3) Train the model  
python model.py

4) Evaluate saved model  
python "load&Evaluate.py"

## Notes
- Dataset paths in `prepareData.py` are hardcoded and may need to be updated.
- Make sure the `Saved/` directory exists before training/evaluation.
