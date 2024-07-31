import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load data
padded_sequences = np.load('cricket_sequences.npy', allow_pickle=True)
print(f"Shape of padded_sequences: {padded_sequences.shape}")

# Create a mask to identify real data vs padding
mask = ~np.all(padded_sequences == 0, axis=2)
print(f"Shape of mask: {mask.shape}")

# Initialize label encoders
le_player = LabelEncoder()
le_batting_style = LabelEncoder()
le_playing_role = LabelEncoder()
le_bowling_style = LabelEncoder()

# Function to encode a column
def encode_column(column, encoder):
    flat_column = column.flatten()
    flat_column = [str(x) for x in flat_column]
    encoded = encoder.fit_transform(flat_column)
    return encoded.reshape(column.shape)

# Encode columns
padded_sequences[:, :, 4] = encode_column(padded_sequences[:, :, 4], le_player)
padded_sequences[:, :, 5] = encode_column(padded_sequences[:, :, 5], le_player)
padded_sequences[:, :, 6] = encode_column(padded_sequences[:, :, 6], le_player)
padded_sequences[:, :, 8] = encode_column(padded_sequences[:, :, 8], le_batting_style)
padded_sequences[:, :, 9] = encode_column(padded_sequences[:, :, 9], le_playing_role)
padded_sequences[:, :, 10] = encode_column(padded_sequences[:, :, 10], le_batting_style)
padded_sequences[:, :, 11] = encode_column(padded_sequences[:, :, 11], le_playing_role)
padded_sequences[:, :, 12] = encode_column(padded_sequences[:, :, 12], le_bowling_style)

# Save encoders
encoders = {
    'player': le_player,
    'batting_style': le_batting_style,
    'playing_role': le_playing_role,
    'bowling_style': le_bowling_style
}
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

# Convert to float
padded_sequences = padded_sequences.astype(float)

# Normalize numerical features
scaler = StandardScaler()
numerical_features = [1, 2, 3, 7, 13, 14, 15, 18]
flat_numerical = padded_sequences[:, :, numerical_features].reshape(-1, len(numerical_features))
normalized_numerical = scaler.fit_transform(flat_numerical)
padded_sequences[:, :, numerical_features] = normalized_numerical.reshape(padded_sequences.shape[0], padded_sequences.shape[1], -1)

with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Convert to torch tensor
X = torch.tensor(padded_sequences[:, :, :-1], dtype=torch.float32)
y = torch.tensor(padded_sequences[:, :, -1], dtype=torch.long)

# Convert runs to classification labels
def runs_to_class(runs):
    runs = int(runs)
    if runs == -1:  # Wicket
        return 8
    return min(runs, 7)

y = torch.tensor([[runs_to_class(r) for r in innings] for innings in y])

# Split the data while maintaining 3D structure
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)




class CricketDataset(Dataset):
    def __init__(self, X, y, mask):
        self.X = X
        self.y = y
        self.mask = mask

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]

class TransformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Ensure input_dim is divisible by num_heads
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.adjusted_dim = self.head_dim * num_heads

        self.attention = nn.MultiheadAttention(self.adjusted_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(self.adjusted_dim)
        self.norm2 = nn.LayerNorm(self.adjusted_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.adjusted_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, self.adjusted_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Add a linear layer to adjust input dimension if necessary
        self.input_proj = nn.Linear(input_dim, self.adjusted_dim) if input_dim != self.adjusted_dim else nn.Identity()

    def forward(self, x):
        x = self.input_proj(x)
        attended, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attended))
        feedforward = self.ff(x)
        return self.norm2(x + self.dropout(feedforward))

# Transformer Model
class CricketTransformer(nn.Module):
    def __init__(self, input_dim, n_players, n_batting_styles, n_playing_roles, n_bowling_styles, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()
        self.player_embedding = nn.Embedding(n_players + 1, 16, padding_idx=n_players)
        self.batting_style_embedding = nn.Embedding(n_batting_styles + 1, 8, padding_idx=n_batting_styles)
        self.playing_role_embedding = nn.Embedding(n_playing_roles + 1, 8, padding_idx=n_playing_roles)
        self.bowling_style_embedding = nn.Embedding(n_bowling_styles + 1, 8, padding_idx=n_bowling_styles)

        self.embedded_dim = input_dim + (16 * 3) + (8 * 2) + (8 * 2) + 8 - 9 + 1
        self.adjusted_dim = ((self.embedded_dim - 1) // num_heads + 1) * num_heads

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.adjusted_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.runs_head = nn.Linear(self.adjusted_dim, 9)  # 0-7 runs + wicket

        self.input_proj = nn.Linear(self.embedded_dim, self.adjusted_dim)

        print(f"Embedded dim: {self.embedded_dim}")
        print(f"Adjusted dim: {self.adjusted_dim}")

    def safe_embedding(self, embedding_layer, indices, num_embeddings):
        safe_indices = torch.clamp(indices, 0, num_embeddings - 1)
        return embedding_layer(safe_indices)

    def forward(self, x):
        player_ids = x[:, :, 4:7].long()
        batting_styles = x[:, :, [8, 10]].long()
        playing_roles = x[:, :, [9, 11]].long()
        bowling_styles = x[:, :, 12].long()

        player_embeddings = self.safe_embedding(self.player_embedding, player_ids, self.player_embedding.num_embeddings)
        batting_style_embeddings = self.safe_embedding(self.batting_style_embedding, batting_styles, self.batting_style_embedding.num_embeddings)
        playing_role_embeddings = self.safe_embedding(self.playing_role_embedding, playing_roles, self.playing_role_embedding.num_embeddings)
        bowling_style_embeddings = self.safe_embedding(self.bowling_style_embedding, bowling_styles, self.bowling_style_embedding.num_embeddings)

        x = torch.cat([
            x[:, :, :4],
            player_embeddings.view(x.size(0), x.size(1), -1),
            x[:, :, 7].unsqueeze(-1),
            batting_style_embeddings.view(x.size(0), x.size(1), -1),
            playing_role_embeddings.view(x.size(0), x.size(1), -1),
            bowling_style_embeddings,
            x[:, :, 13:]
        ], dim=2)

        x = self.input_proj(x)

        for transformer in self.transformer_blocks:
            x = transformer(x)

        runs = self.runs_head(x[:, -1, :])  # Use last token for prediction
        return runs


# Training function
# Modify the train function to handle 3D data and masks
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            features, targets, batch_mask = [b.to(device) for b in batch]
            optimizer.zero_grad()
            runs_pred = model(features)
            loss = criterion(runs_pred.view(-1, 9), targets[batch_mask].view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                features, targets, batch_mask = [b.to(device) for b in batch]
                runs_pred = model(features)
                loss = criterion(runs_pred.view(-1, 9), targets[batch_mask].view(-1))
                val_loss += loss.item()

                _, predicted = torch.max(runs_pred.view(-1, 9), 1)
                total += targets[batch_mask].view(-1).size(0)
                correct += (predicted == targets[batch_mask].view(-1)).sum().item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return model

# Main training loop
def print_sample_predictions(model, val_loader, n_samples=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Get a batch of validation data
    features, targets = next(iter(val_loader))
    features, targets = features.to(device), targets.to(device)

    with torch.no_grad():
        predictions = model(features)

    # Convert predictions to probabilities
    probabilities = torch.nn.functional.softmax(predictions, dim=1)

    print("\nSample Predictions:")
    print("------------------")
    for i in range(min(n_samples, len(targets))):
        true_runs = targets[i, -1].item()

        if true_runs == 8:
            true_label = 'W'
        else:
            true_label = str(true_runs)

        print(f"Sample {i+1}:")
        print(f"  True runs: {true_label}")
        print("  Predicted probabilities:")
        for j in range(9):  # 0-7 runs + wicket
            if j == 8:
                run_label = 'W'
            else:
                run_label = str(j)
            print(f"    {run_label}: {probabilities[i, j]:.4f}")
        print()


def main():
    train_mask = ~torch.all(X_train == 0, dim=2)
    test_mask = ~torch.all(X_test == 0, dim=2)
    
    train_dataset = CricketDataset(X_train, y_train, train_mask)
    test_dataset = CricketDataset(X_test, y_test, test_mask)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32)

    input_dim = X_train.shape[2]  # Use the actual input dimension
    print(f"Input dimension: {input_dim}")

    num_heads = 8  # You can adjust this value

    n_players = len(le_player.classes_)
    n_batting_styles = len(le_batting_style.classes_)
    n_playing_roles = len(le_playing_role.classes_)
    n_bowling_styles = len(le_bowling_style.classes_)

    print(f"Number of unique players: {n_players}")
    print(f"Number of unique batting styles: {n_batting_styles}")
    print(f"Number of unique playing roles: {n_playing_roles}")
    print(f"Number of unique bowling styles: {n_bowling_styles}")

    model = CricketTransformer(
        input_dim=input_dim,
        n_players=n_players,
        n_batting_styles=n_batting_styles,
        n_playing_roles=n_playing_roles,
        n_bowling_styles=n_bowling_styles,
        num_heads=8,
        num_layers=6,
        ff_dim=256,
        dropout=0.1
    )

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = train(model, train_loader, val_loader, optimizer, criterion, num_epochs=50)

    # Save the model
    torch.save(model.state_dict(), 'cricket_transformer_model.pth')

    print_sample_predictions(model, val_loader)

if __name__ == "__main__":
    main()