#inspiration for the model: https://github.com/eeyhsong/EEG-Conformer/tree/main
#Data: https://www.bbci.de/competition/iv/ Data sets 2a, description: https://www.bbci.de/competition/iv/desc_2a.pdf

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter


X_t = np.load(r"C:\Users\Helene\Desktop\Uni\Advanced Deep Learning\Projekt\Data\Kompakt\X_bci_2a_training_data.npy")
y_t = np.load(r"C:\Users\Helene\Desktop\Uni\Advanced Deep Learning\Projekt\Data\Kompakt\y_bci_2a_training_data.npy")

#Convert Labels --> int
y_t = y_t.astype(int)

#Map Event IDs to Classes
event_mapping = {769: 0, 770: 1, 771: 2, 772: 3}


valid_indices_t = [i for i, event in enumerate(y_t) if event in event_mapping]
y_t = np.array([event_mapping[y_t[i]] for i in valid_indices_t])
X_t = X_t[valid_indices_t]

#80/20 Split
X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.2, random_state=42, stratify=y_t)

#Debug
print("Shape von X_train:", X_train.shape)
print("Shape von y_train:", y_train.shape)
print("Shape von X_val:", X_val.shape)
print("Shape von y_val:", y_val.shape)
print("Train Class Distribution:", Counter(y_train))
print("Val Class Distribution:", Counter(y_val))


class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

#Dataloader
batch_size = 32
train_dataset = EEGDataset(X_train, y_train)
val_dataset = EEGDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#Model Structure
#Convolution module: Capturing temporal and spatial features from EEGsignals
class ConvModule(nn.Module):
    def __init__(self, num_channels, k=40, kernel_size=(1, 25), pooling_size=(1, 75), pooling_stride=(1, 15)):
        super(ConvModule, self).__init__()
        self.temporal_conv = nn.Conv2d(1, k, kernel_size=kernel_size, stride=(1, 1))
        self.spatial_conv = nn.Conv2d(k, k, kernel_size=(num_channels, 1), stride=(1, 1))
        self.batch_norm = nn.BatchNorm2d(k)
        self.activation = nn.ELU()
        self.pooling = nn.AvgPool2d(kernel_size=pooling_size, stride=pooling_stride)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Temporal convolution -> Spatial convolution -> Pooling -> Dropout
        x = self.temporal_conv(x)
        x = self.activation(self.batch_norm(x))
        x = self.spatial_conv(x)
        x = self.activation(self.batch_norm(x))
        x = self.pooling(x)
        x = self.dropout(x)
        x = x.squeeze(2).permute(0, 2, 1)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attention = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        attention = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention, V).transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.dropout(self.out(output))

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.mha(x))
        x = self.norm2(x + self.ff(x))
        return x
#Self-attention module: Extracting global temporal dependencies
class SelfAttentionModule(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(SelfAttentionModule, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
#Final classifier: Mapping learned features to class probabilities
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(x)

#Final model: Combines convolution, self-attention, and classifier modules
class EEGModel(nn.Module):
    def __init__(self, num_channels, d_model, num_classes, num_heads, num_layers, d_ff):
        super(EEGModel, self).__init__()
        self.conv_module = ConvModule(num_channels, d_model)
        self.self_attention = SelfAttentionModule(d_model, num_heads, d_ff, num_layers)
        self.classifier = Classifier(input_dim=d_model, num_classes=num_classes)

    def forward(self, x):
        x = self.conv_module(x)
        x = self.self_attention(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

#Hyperparameters final
num_channels = X_train.shape[2]
d_model = 40
num_classes = len(set(y_train))
num_heads = 4
num_layers = 2
d_ff = 64

model = EEGModel(num_channels, d_model, num_classes, num_heads, num_layers, d_ff)

#Loss, optimizer and lr scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

#Training with early stopping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
epochs = 100
early_stop_patience = 10
min_val_loss = float("inf")
early_stop_counter = 0

train_loss_history, val_loss_history = [], []
train_accuracy_history, val_accuracy_history = [], []

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        #Forward pass --> Compute loss --> Backpropagation --> Optimization step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = correct / total
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    #Validation
    model.eval()
    val_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = correct / total
    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    scheduler.step(val_loss)

    #Early Stopping
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

#confusion matrix and classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

report = classification_report(all_labels, all_preds, target_names=["Class 0", "Class 1", "Class 2", "Class 3"])
print("Classification Report:")
print(report)

#Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Validation Loss", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracy_history, label="Train Accuracy")
plt.plot(val_accuracy_history, label="Validation Accuracy", linestyle="--")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Over Epochs")
plt.legend()

plt.tight_layout()
plt.show()


#Modelsave

model_save_path = r"C:\Users\Helene\Desktop\Uni\Advanced Deep Learning\Projekt\Final\EEGModel.pth"


torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")


#Inference
#
model_path = r"C:\Users\Helene\Desktop\Uni\Advanced Deep Learning\Projekt\Final\EEGModel.pth"
X_eval_path = r"C:\Users\Helene\Desktop\Uni\Advanced Deep Learning\Projekt\Data\Kompakt\X_bci_2a_evaluation_data_2.npy"
y_eval_path = r"C:\Users\Helene\Desktop\Uni\Advanced Deep Learning\Projekt\Data\Kompakt\y_bci_2a_evaluation_data_2.npy"

X_eval = np.load(X_eval_path)
y_eval = np.load(y_eval_path).astype(int)

class EEGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 32
eval_dataset = EEGDataset(X_eval, y_eval)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EEGModel(num_channels, d_model, num_classes, num_heads, num_layers, d_ff)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

#Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in eval_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


event_mapping = {769: 0, 770: 1, 771: 2, 772: 3}
all_labels = np.array([event_mapping[label] for label in all_labels])

#Debug
print("Unique labels in ground truth (after mapping):", np.unique(all_labels))
print("Unique predictions:", np.unique(all_preds))


num_classes = len(np.unique(all_labels))  
target_names = [f"Class {i}" for i in range(num_classes)]  

#Confusion matrix und classification report
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=target_names)


print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

