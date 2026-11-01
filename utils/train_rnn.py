import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import joblib
from pathlib import Path

MODEL_DIR = Path("/models")

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.dropout(out[:, -1, :])  # Take last timestep
        out = self.relu(self.fc1(out))
        out = self.softmax(self.fc2(out))
        return out

def train_rnn(X_train, X_test, y_train, y_test):
    # Convert to tensors (handle both numpy arrays and pandas Series)
    import numpy as np
    
    X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
    X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
    y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    
    X_train_tensor = torch.tensor(X_train_np.reshape(X_train_np.shape[0], X_train_np.shape[1], 1), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np.reshape(X_test_np.shape[0], X_test_np.shape[1], 1), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np.astype(int), dtype=torch.long)
    y_test_tensor = torch.tensor(y_test_np.astype(int), dtype=torch.long)

    n_classes = len(np.unique(y_train_np))

    model = RNNModel(1, 64, n_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(20):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "rnn.pth")

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs = model(X_test_tensor).numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs