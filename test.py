# ============================================================
# Import der Bibliotheken
# ============================================================
import urllib.request  # Lädt den Iris-Datensatz direkt aus dem Internet
import io              # Macht aus geladenen Bytes/String ein Dateiobjekt
import pandas as pd    # Zum Einlesen und Bearbeiten des CSV-Datensatzes
import numpy as np     # N-dimensionale Arrays und numerische Operationen
import torch           # Hauptpaket von PyTorch: Tensors, Training, GPU
from torch import nn   # Bausteine für neuronale Netze (Layer, Loss etc.)
from torch.utils.data import TensorDataset, DataLoader  # Daten für Training/Test verwalten
from sklearn.model_selection import train_test_split    # Train/Test-Split
from sklearn.preprocessing import StandardScaler        # Daten normalisieren
from sklearn.metrics import confusion_matrix, classification_report  # Auswertung

# ============================================================
# 1) Iris-Datensatz laden
# ------------------------------------------------------------
# UCI Machine Learning Repository hat den Datensatz im CSV-Format.
# Wir holen ihn als Text und lesen ihn mit pandas ein.
# ============================================================
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

with urllib.request.urlopen(URL) as resp:      # Download
    raw = resp.read().decode("utf-8")          # Bytes -> String

df = pd.read_csv(io.StringIO(raw), header=None, names=cols)  # CSV in DataFrame
df = df.dropna()  # Falls leere Zeilen vorhanden sind

print("Load Done")