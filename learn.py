1# === 🧠 1. 導入套件 ===
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# === ⚙️ 2. 函數：找出爆發點 ===
def find_positive_indices(df, price_col='close', future_window=120, threshold=0.05):
    indices = []
    prices = df[price_col].values
    for i in range(len(prices) - future_window):
        future_max = prices[i+1:i+future_window+1].max()
        future_return = (future_max - prices[i]) / prices[i]
        if future_return >= threshold:
            indices.append(i)
    return indices

# === ⚙️ 3. 函數：建立樣本（滑動視窗） ===
def build_sequences_sliding(df, positive_idx, feature_cols, seq_len=480, future_window=120, neg_ratio=1.0, slide_range=10):
    pos_samples = []
    neg_samples = []

    for i in positive_idx:
        for shift in range(slide_range):
            idx = i - shift
            if idx - seq_len >= 0:
                seq = df.iloc[idx - seq_len:idx]
                pos_samples.append((seq[feature_cols].values, 1))

    all_possible = set(range(seq_len, len(df) - future_window))
    exclude = set()
    for i in positive_idx:
        exclude.update(range(i - slide_range - 1, i + future_window + 1))
    neg_candidates = list(all_possible - exclude)

    np.random.shuffle(neg_candidates)
    neg_limit = int(len(pos_samples) * neg_ratio)
    for i in neg_candidates[:neg_limit]:
        seq = df.iloc[i - seq_len:i]
        neg_samples.append((seq[feature_cols].values, 0))

    samples = pos_samples + neg_samples
    np.random.shuffle(samples)
    return samples

# === ⚙️ 4. Dataset 類別 ===
class SegmentDataset(Dataset):
    def __init__(self, samples):
        self.X = [torch.tensor(x, dtype=torch.float32) for x, _ in samples]
        self.y = [torch.tensor(y, dtype=torch.long) for _, y in samples]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# === ⚙️ 5. 強化版 LSTM 模型 ===
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        return self.fc(out)

# === ⚙️ 6. 強化版訓練流程 ===
def train_model(samples, input_size, batch_size=64, epochs=30, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentDataset(samples)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    class_weight = torch.tensor([1.0, 1.5], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    model = LSTMModel(input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)

    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)

    test_loader = DataLoader(val_ds, batch_size=128)
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu()
            y_pred.extend(preds.numpy())
            y_true.extend(y.numpy())

    print("\n\U0001F4CA Test Report:\n")
    print(classification_report(y_true, y_pred))

    return model

# === 🚀 主流程 ===
df = pd.read_csv("ETHUSDT_1m_klines_merged.csv")
df.columns = df.columns.str.lower().str.strip()
df['open_time'] = pd.to_datetime(df['open_time'])

feature_cols = ['open', 'high', 'low', 'close', 'volume']
scaler = StandardScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

positive_idx = find_positive_indices(df, price_col='close', future_window=120, threshold=0.05)
print(f"爆發點數量: {len(positive_idx)}")

samples = build_sequences_sliding(df, positive_idx, feature_cols, seq_len=480, future_window=120, neg_ratio=1.0, slide_range=10)
print(f"樣本數量: {len(samples)}")

model = train_model(samples, input_size=len(feature_cols))
torch.save(model.state_dict(), "lstm_explosive_model.pt")
print("✅ 模型儲存成功")
