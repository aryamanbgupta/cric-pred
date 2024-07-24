import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, LayerNormalization, MultiHeadAttention, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the data
sequences = np.load('cricket_sequences.npy', allow_pickle=True)

# Preprocess the data
max_sequence_length = max(len(seq) for seq in sequences)
n_features = len(sequences[0][0])

# Pad sequences
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_sequence_length, padding='post', dtype='float32'
)

# Separate features and target
X = padded_sequences[:, :, :-1]  # All columns except the last
y = padded_sequences[:, :, -1]   # Last column is the runs scored

# Convert string IDs to numeric
le_batter = LabelEncoder()
le_bowler = LabelEncoder()

X[:, :, 4] = le_batter.fit_transform(X[:, :, 4].flatten()).reshape(X[:, :, 4].shape)
X[:, :, 5] = le_bowler.fit_transform(X[:, :, 5].flatten()).reshape(X[:, :, 5].shape)

n_batters = len(le_batter.classes_)
n_bowlers = len(le_bowler.classes_)
n_players = max(n_batters, n_bowlers)

# Normalize numerical features
scaler = StandardScaler()
X[:, :, [1, 2, 3, 6]] = scaler.fit_transform(X[:, :, [1, 2, 3, 6]].reshape(-1, 4)).reshape(X[:, :, [1, 2, 3, 6]].shape)

# Convert runs to classification labels
def runs_to_class(runs):
    if runs == 'W':
        return 8
    runs = int(float(runs))
    return min(runs, 7)

y = np.vectorize(runs_to_class)(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformer block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

# Model architecture
inputs = Input(shape=(max_sequence_length, n_features - 1))

# Embedding layer for players
player_embedding = Embedding(n_players, 16)
batter_embed = player_embedding(inputs[:, :, 4])
bowler_embed = player_embedding(inputs[:, :, 5])

# Concatenate all inputs
x = tf.keras.layers.Concatenate()([
    inputs[:, :, :4],
    batter_embed,
    bowler_embed,
    inputs[:, :, 6:]
])

# Transformer layers
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=32, dropout=0.1)
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=32, dropout=0.1)

# Output layer
outputs = Dense(9, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

model.save('transformer_pred_v1.h5')
encoders = {
    'batter': le_batter,
    'bowler': le_bowler
}

with open('transformer_label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print(f"Test accuracy: {test_accuracy:.4f}")

# Function to predict
def predict_next_ball(model, input_sequence):
    prediction = model.predict(input_sequence)
    return prediction[0][-1]  # Return prediction for the last ball in the sequence

# Example prediction
sample_input = X_test[0:1]
prediction = predict_next_ball(model, sample_input)
print("Predicted probabilities:")
for i, prob in enumerate(prediction):
    if i < 8:
        print(f"{i} runs: {prob:.4f}")
    else:
        print(f"Wicket: {prob:.4f}")