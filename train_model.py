import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data
from model_training import train_and_evaluate_model
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Embedding
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, confusion_matrix
import seaborn as sns

def plot_metrics(history, y_test, y_pred, y_pred_proba):
    """Plot training history and evaluation metrics"""
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot training history
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot ROC curve
    plt.subplot(2, 2, 3)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    # Plot confusion matrix
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../results/model_evaluation.png')
    plt.close()

def main():
    # Create directories if they don't exist
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../results', exist_ok=True)

    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_protein, X_drug, y = load_and_preprocess_data('../data/raw/disease.csv')

    # Train model
    print("Training model...")
    model, metrics = train_and_evaluate_model(X_protein, X_drug, y)

    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nModel saved to: ../models/drug_protein_model.h5")

if __name__ == "__main__":
    main()