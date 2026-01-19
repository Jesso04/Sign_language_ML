import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Load data
X = np.load("X.npy")
y = np.load("y.npy")

# Flatten landmarks per frame: (60, 21, 3) → (60, 63)
X = X.reshape(X.shape[0], X.shape[1], -1)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Model definition
class SignLSTM(nn.Module):
    def __init__(self, input_size=63, hidden_size=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model = SignLSTM()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        val_acc = (val_outputs.argmax(1) == y_val).float().mean()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {loss.item():.4f} | "
        f"Val Loss: {val_loss.item():.4f} | "
        f"Val Acc: {val_acc:.2%}"
    )

# Save model
torch.save(model.state_dict(), "sign_model.pth")
print("Model saved as sign_model.pth")
