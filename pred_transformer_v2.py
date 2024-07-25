import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load data
sequences = np.load('cricket_sequences.npy', allow_pickle=True)

# Function to convert variable length sequences to fixed length
def pad_sequences(sequences, max_len=None):
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    # Determine the number of features
    n_features = sequences[0].shape[1]

    # Initialize padded sequences with empty strings for object dtype
    padded_sequences = np.full((len(sequences), max_len, n_features), '', dtype=object)

    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq

    return padded_sequences

# Pad sequences
padded_sequences = pad_sequences(sequences)

# Initialize label encoders
le_player = LabelEncoder()
le_batting_style = LabelEncoder()
le_playing_role = LabelEncoder()
le_bowling_style = LabelEncoder()

# Function to encode a column
def encode_column(column, encoder):
    flat_column = column.flatten()
    # Convert all values to strings
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

n_players = len(le_player.classes_)
n_batting_styles = len(le_batting_style.classes_)
n_playing_roles = len(le_playing_role.classes_)
n_bowling_styles = len(le_bowling_style.classes_)

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

# Convert to torch tensor
X = torch.tensor(padded_sequences, dtype=torch.float32)

# Separate features and target
y = X[:, :, -1].long()  # Last column is the runs scored
X = X[:, :, :-1]  # All columns except the last

# Normalize numerical features
scaler = StandardScaler()
numerical_features = [1, 2, 3, 7, 13, 14, 15, 18]
X[:, :, numerical_features] = torch.tensor(
    scaler.fit_transform(X[:, :, numerical_features].reshape(-1, len(numerical_features)).numpy())
).reshape(X[:, :, numerical_features].shape)

with open('standard_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

# Convert runs to classification labels
def runs_to_class(runs):
    if runs == 'W':
        return 8
    runs = int(float(runs))
    return min(runs, 7)

y = torch.tensor(np.vectorize(runs_to_class)(y.numpy()))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# The rest of the code (CricketDataset, CricketTransformer, etc.) remains the same

# Custom Dataset
class CricketDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        self.player_embedding = nn.Embedding(n_players + 1, 16, padding_idx=n_players)  # +1 for unknown players
        self.batting_style_embedding = nn.Embedding(n_batting_styles + 1, 8, padding_idx=n_batting_styles)
        self.playing_role_embedding = nn.Embedding(n_playing_roles + 1, 8, padding_idx=n_playing_roles)
        self.bowling_style_embedding = nn.Embedding(n_bowling_styles + 1, 8, padding_idx=n_bowling_styles)

        # Calculate the dimension after embeddings
        self.embedded_dim = input_dim + (16 * 3) + (8 * 2) + (8 * 2) + 8 - 9 + 1 # 9 is the number of features we're replacing with embeddings

        # Adjust embedded_dim to be divisible by num_heads
        self.adjusted_dim = ((self.embedded_dim - 1) // num_heads + 1) * num_heads

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.adjusted_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.runs_head = nn.Linear(self.adjusted_dim, 9)  # 0-7 runs + wicket
        self.noball_head = nn.Linear(self.adjusted_dim, 2)
        self.wide_noball_head = nn.Linear(self.adjusted_dim, 2)

        # Add a linear layer to adjust input dimension
        self.input_proj = nn.Linear(self.embedded_dim, self.adjusted_dim)

        print(f"Embedded dim: {self.embedded_dim}")
        print(f"Adjusted dim: {self.adjusted_dim}")

    def safe_embedding(self, embedding_layer, indices, num_embeddings):
        # Clip indices to be within the valid range
        safe_indices = torch.clamp(indices, 0, num_embeddings - 1)
        return embedding_layer(safe_indices)

    def forward(self, x):
        #print(f"Input shape: {x.shape}")

        # Extract features for embedding
        player_ids = x[:, :, 4:7].long()
        batting_styles = x[:, :, [8, 10]].long()
        playing_roles = x[:, :, [9, 11]].long()
        bowling_styles = x[:, :, 12].long()

        # Apply embeddings with safety check
        player_embeddings = self.safe_embedding(self.player_embedding, player_ids, self.player_embedding.num_embeddings)
        batting_style_embeddings = self.safe_embedding(self.batting_style_embedding, batting_styles, self.batting_style_embedding.num_embeddings)
        playing_role_embeddings = self.safe_embedding(self.playing_role_embedding, playing_roles, self.playing_role_embedding.num_embeddings)
        bowling_style_embeddings = self.safe_embedding(self.bowling_style_embedding, bowling_styles, self.bowling_style_embedding.num_embeddings)

        # Concatenate embeddings with other features
        x = torch.cat([
            x[:, :, :4],
            player_embeddings.view(x.size(0), x.size(1), -1),
            x[:, :, 7].unsqueeze(-1),
            batting_style_embeddings.view(x.size(0), x.size(1), -1),
            playing_role_embeddings.view(x.size(0), x.size(1), -1),
            bowling_style_embeddings,
            x[:, :, 13:]
        ], dim=2)

        #print(f"After embedding shape: {x.shape}")

        x = self.input_proj(x)
        #print(f"After input projection shape: {x.shape}")

        for transformer in self.transformer_blocks:
            x = transformer(x)

        #print(f"After transformer blocks shape: {x.shape}")

        runs = self.runs_head(x[:, -1, :])  # Use last token for prediction
        noball = self.noball_head(x[:, -1, :])
        wide_noball = self.wide_noball_head(x[:, -1, :])
        return runs, noball, wide_noball

# Training function
def train(model, train_loader, val_loader, optimizer, criterion, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            features, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            runs_pred, noball_pred, wide_noball_pred = model(features)
            loss = (criterion(runs_pred, targets[:, -1]) +
                    criterion(noball_pred, features[:, -1, 16].long()) +
                    criterion(wide_noball_pred, features[:, -1, 17].long()))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features, targets = [b.to(device) for b in batch]
                runs_pred, noball_pred, wide_noball_pred = model(features)
                loss = (criterion(runs_pred, targets[:, -1]) +
                        criterion(noball_pred, features[:, -1, 16].long()) +
                        criterion(wide_noball_pred, features[:, -1, 17].long()))
                val_loss += loss.item()

        print(f"Epoch {epoch+1}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

# Main training loop
def main():
    train_dataset = CricketDataset(X_train, y_train)
    test_dataset = CricketDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=32)

    # Correctly calculate input_dim by adding numerical features instead of subtracting
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
        num_heads=num_heads,
        num_layers=6,
        ff_dim=256
    )

    print(f"Embedded dim: {model.embedded_dim}")
    print(f"Adjusted dim: {model.adjusted_dim}")
    print(f"Model structure:\n{model}")

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, val_loader, optimizer, criterion, num_epochs=50)

    # Save the model
    torch.save(model.state_dict(), 'cricket_transformer.pth')

if __name__ == "__main__":
    main()