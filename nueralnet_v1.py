import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load the data
data = np.load('cricket_training_data.npy', allow_pickle=True)

print(f"Loaded data shape: {data.shape}")
print("Sample data:")
print(data[:5])

# Separate features and target
X = data[:, :-1]  # All columns except the last
y = data[:, -1]   # Last column is the runs scored

print("X shape:", X.shape)
print("y shape:", y.shape)

# Convert string IDs to numeric
le_batter = LabelEncoder()
le_bowler = LabelEncoder()

X[:, 4] = le_batter.fit_transform(X[:, 4])
X[:, 5] = le_bowler.fit_transform(X[:, 5])

# Now we can safely get the number of unique players
n_batters = len(le_batter.classes_)
n_bowlers = len(le_bowler.classes_)
n_players = max(n_batters, n_bowlers)

print(f"Number of unique batters: {n_batters}")
print(f"Number of unique bowlers: {n_bowlers}")
print(f"Total number of players: {n_players}")

# Ensure all columns in X are numeric
X = X.astype(float)

# Normalize numerical features
scaler = StandardScaler()
X[:, [1, 2, 3, 6]] = scaler.fit_transform(X[:, [1, 2, 3, 6]])

# Convert runs to classification labels
def runs_to_class(runs):
    if runs == 'W':  # Assuming 'W' represents a wicket
        return 8
    runs = int(float(runs))  # Convert to float first in case of decimal values
    return min(runs, 7)  # 0-6 runs mapped to 0-6, 7+ runs mapped to 7

y = np.vectorize(runs_to_class)(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define input layers
inning_input = Input(shape=(1,))
score_input = Input(shape=(1,))
wickets_input = Input(shape=(1,))
balls_input = Input(shape=(1,))
batter_input = Input(shape=(1,))
bowler_input = Input(shape=(1,))
position_input = Input(shape=(1,))

# Embedding layer for players
player_embedding = Embedding(n_players, 16)
batter_embed = Flatten()(player_embedding(batter_input))
bowler_embed = Flatten()(player_embedding(bowler_input))

# Concatenate all inputs
x = Concatenate()([inning_input, score_input, wickets_input, balls_input, 
                   batter_embed, bowler_embed, position_input])

# Hidden layers
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

# Output layer
output = Dense(9, activation='softmax')(x)  # 9 classes: 0-7 runs and wicket

# Create the model
model = Model(inputs=[inning_input, score_input, wickets_input, balls_input, 
                      batter_input, bowler_input, position_input], 
              outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    [X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3], 
     X_train[:, 4], X_train[:, 5], X_train[:, 6]],
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(
    [X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3], 
     X_test[:, 4], X_test[:, 5], X_test[:, 6]],
    y_test,
    verbose=0
)

model.save('pred_v1.h5')
encoders = {
    'batter': le_batter,
    'bowler': le_bowler
}

with open('label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)
    
print(f"Test accuracy: {test_accuracy:.4f}")

# Function to predict
def predict_next_ball(model, input_data):
    prediction = model.predict([input_data[0:1], input_data[1:2], input_data[2:3], 
                                input_data[3:4], input_data[4:5], input_data[5:6], input_data[6:7]])
    return prediction[0]

# Example prediction
sample_input = X_test[0]
prediction = predict_next_ball(model, sample_input)
print("Predicted probabilities:")
for i, prob in enumerate(prediction):
    if i < 8:
        print(f"{i} runs: {prob:.4f}")
    else:
        print(f"Wicket: {prob:.4f}")