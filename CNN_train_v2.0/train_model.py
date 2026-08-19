import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==========================================
# Configuration & Hardware Setup
# ==========================================
# Automatically detect if your NVIDIA GPU is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {DEVICE}")

EPOCHS = 30
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_CLASSES = 4  

# ==========================================
# 1. Dataset Memory Management
# ==========================================
class SensorDataset(Dataset):
    def __init__(self, x_path, y_path):
        # Load the raw C-contiguous arrays from disk
        X_numpy = np.load(x_path)
        y_numpy = np.load(y_path)
        
        # PyTorch 1D-CNNs expect shape: [Batch, Channels, Timesteps]
        # Our CSV made [Batch, Timesteps, Channels] (50x10). We must transpose it to 10x50.
        X_numpy = np.transpose(X_numpy, (0, 2, 1))
        
        # Allocate into PyTorch Tensors
        self.X = torch.tensor(X_numpy, dtype=torch.float32)
        self.y = torch.tensor(y_numpy, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. The Neural Network Blueprint
# ==========================================
class GestureNet(nn.Module):
    def __init__(self, num_classes):
        super(GestureNet, self).__init__()
        
        # Feature Extraction layer (Sliding the filters)
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # Compresses 50 timesteps down to 25
            
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Compresses 25 timesteps down to 12
        )
        
        # Decision layer (The final classification)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12, 128),     # 64 channels * 12 remaining timesteps
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.conv_block(x)
        output = self.classifier(features)
        return output

# ==========================================
# 3. The Execution Engine
# ==========================================
def main():
    # Load memory blocks and slice into batches
    dataset = SensorDataset("training_tensors_data/X_train.npy", "training_tensors_data/y_train.npy")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize the network and push it to the GPU
    model = GestureNet(NUM_CLASSES).to(DEVICE)
    
    # Math functions for calculating error and updating weights
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting Training Loop...")
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct_predictions = 0
        
        for batch_X, batch_y in dataloader:
            # Push the data chunk to the GPU memory
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            # Step 1: Forward Pass (Make a guess)
            optimizer.zero_grad() 
            predictions = model(batch_X)
            
            # Step 2: Calculate Error
            loss = criterion(predictions, batch_y)
            
            # Step 3: Backpropagation (Calculate gradients)
            loss.backward()
            
            # Step 4: Update internal weights
            optimizer.step()
            
            # Track accuracy
            total_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_predictions += (predicted_classes == batch_y).sum().item()
            
        epoch_accuracy = (correct_predictions / len(dataset)) * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")
        
    # Save the compiled weights to disk
    torch.save(model.state_dict(), "gesture_model.pth")
    print("\nTraining Complete! Model saved as 'gesture_model.pth'")

if __name__ == "__main__":
    main()