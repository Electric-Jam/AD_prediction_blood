import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

from pydeseq2.preprocessing import deseq2_norm

#Input processing functions
def load_data(meta_path, rna_seq_path, id_col, label_col, normalize=True, run_pca=False, pca_n=10, pca_model=None, gene_list=None):
    meta_data = _load_metadata(meta_path, id_col, label_col)
    samples_to_keep = ~meta_data.AD.isna()
    meta_data = meta_data.loc[samples_to_keep]

    rna_seq_data = _load_rna_seq(rna_seq_path)
    rna_seq_data = rna_seq_data.loc[samples_to_keep]

    if normalize:
        norm_rna_seq_data = normalize_RNAseq(rna_seq_data)[0]
    else:
        norm_rna_seq_data = rna_seq_data

    # Filter genes
    if gene_list is not None:
        rna_seq_data = rna_seq_data.loc[:, gene_list]

    if run_pca:
        pca = PCA(n_components=pca_n)
        pca_data = pca.fit_transform(norm_rna_seq_data)
        norm_rna_seq_data = pd.DataFrame(pca_data, index=norm_rna_seq_data.index)

    if pca_model:
        pca_data = pca_model.transform(norm_rna_seq_data)
        norm_rna_seq_data = pd.DataFrame(pca_data, index=norm_rna_seq_data.index)
        pca = None

    merged_data = pd.merge(meta_data, norm_rna_seq_data, left_index=True, right_index=True)

    return merged_data, pca

def _load_metadata(meta_path, id_col, label_col):
    meta_data = pd.read_excel(meta_path)
    meta_data = meta_data.set_index(id_col)

    # Specify label conversion before label encoding
    meta_data[label_col] = meta_data[label_col].replace({'AD': 1, 'yes': 1, 'N': 0, 'no': 0, 'NCI': 0})
    meta_data = meta_data.rename(columns={label_col: 'AD'})

    return meta_data[['AD']]

def _load_rna_seq(rna_seq_path):
    rnaseq_data = pd.read_csv(rna_seq_path, sep='\t', index_col=0)
    rna_seq_data = rnaseq_data.T
    return rna_seq_data

def normalize_RNAseq(rna_seq_df):
    normalized_rnaseq = deseq2_norm(rna_seq_df)

    return normalized_rnaseq

# Classifying functions
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 5)
        self.fc2 = nn.Linear(5, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

class PyTorchClassifier:
    def __init__(self, input_dim, epochs=100, batch_size=10, lr=0.001):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = SimpleNN(input_dim)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        best_loss = float('inf')
        best_model = None
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            for data, target in dataloader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            self.model.eval()
            with torch.no_grad():
                output = self.model(X)
                predictions = (output > 0.5).float()
                f1 = f1_score(y, predictions)
            
            if (epoch + 1) % 20 == 0 or epoch == self.epochs - 1:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {epoch_loss:.4f}, F1 Score: {f1:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = self.model.state_dict()
        
        self.model.load_state_dict(best_model)
        torch.save(self.model.state_dict(), 'best_model.pth')
        
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            predictions = (output > 0.5).float()
        return predictions.numpy()
    
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
        return output.numpy()

def plot_AUC(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_CM(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

def Classifier(data, best_model_path='best_model.pth', epochs=100, batch_size=10, lr=0.001, num_features=10):
    X = data.iloc[:, 1:num_features+1].values  # Extracting the specified number of features
    y = data.iloc[:, 0].values  # Assuming column 0 is the target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_loss = float('inf')
    best_model_state_dict = None
    X_test_best = None
    y_test_best = None

    for fold, (train_index, test_index) in enumerate(skf.split(X_scaled, y)):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = PyTorchClassifier(input_dim=num_features, epochs=epochs, batch_size=batch_size, lr=lr)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        proba_predictions = clf.predict_proba(X_test)
        loss = clf.criterion(torch.tensor(proba_predictions, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)).item()
        print(f"Fold {fold+1}, Loss: {loss:.4f}")

        if loss < best_loss:
            best_loss = loss
            best_model_state_dict = clf.model.state_dict()
            X_test_best = X_test
            y_test_best = y_test

    if best_model_state_dict:
        torch.save(best_model_state_dict, best_model_path)

        # Plotting AUC and Confusion Matrix for the best model
        best_model = PyTorchClassifier(input_dim=num_features, epochs=epochs, batch_size=batch_size, lr=lr)
        best_model.model.load_state_dict(best_model_state_dict)
        best_model.model.eval()
        proba_predictions_best = best_model.predict_proba(X_test_best)
        predictions_best = best_model.predict(X_test_best)
        
        plot_AUC(y_test_best, proba_predictions_best)
        plot_CM(y_test_best, predictions_best)
    
    return best_model_path

# Applying to another dataset
def apply_classifier(data, model_path, num_features=10):
    X = data.iloc[:, 1:num_features+1].values
    y = data.iloc[:, 0].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = PyTorchClassifier(input_dim=num_features)
    clf.model.load_state_dict(torch.load(model_path))
    predictions = clf.predict(X_scaled)
    proba_predictions = clf.predict_proba(X_scaled)

    f1 = f1_score(y, predictions)
    print(f"F1 Score: {f1:.4f}")

    plot_AUC(y, proba_predictions)
    plot_CM(y, predictions)

    return f1