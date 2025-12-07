import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# Dummy training data: [EAR, yawn_distance], 100 samples, 30 time steps
X = np.random.rand(100, 30, 2)  # shape: (samples, time_steps, features)
y = np.random.randint(0, 2, size=(100, 1))  # Binary label: drowsy (1) or not (0)

# Define the model
model = Sequential()
model.add(LSTM(64, return_sequences=False, input_shape=(30, 2)))  # (time_steps, features)
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=3, batch_size=8)

# Save the model
model.save("adaptive_threshold_model_2features.h5")
print("✅ 2-feature model saved as adaptive_threshold_model_2features.h5")
