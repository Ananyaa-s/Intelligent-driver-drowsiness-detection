import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Load actual data from your CSV
data = pd.read_csv("user_data.csv")

# Use only necessary columns
features = data[['ear', 'yawn_distance']].values

# Label based on thresholds
ear_threshold = data['ear_threshold'].iloc[0]
yawn_threshold = data['yawn_threshold'].iloc[0]

# Label: 1 = Drowsy, 0 = Alert
labels = ((data['ear'] < ear_threshold) & (data['yawn_distance'] > yawn_threshold)).astype(int).values

# Build sequences (30 timesteps each)
sequence_length = 30
X, y = [], []
for i in range(len(features) - sequence_length):
    X.append(features[i:i+sequence_length])
    y.append(labels[i+sequence_length - 1])  # Label based on the last timestep

X = np.array(X)
y = np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential()
model.add(LSTM(64, input_shape=(30, 2)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save('snapawake_lstm_3features.h5')