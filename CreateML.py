import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def generate_data():
    # Data acak dengan 1000 sampel, 10 fitur
    X = np.random.rand(1000, 10)
    # Label biner (0 atau 1)
    y = np.random.randint(0, 2, 1000)
    return X, y

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(64, input_shape=(input_shape,), activation='relu'))  # Layer input
    model.add(Dense(32, activation='relu'))  # Hidden layer
    model.add(Dense(1, activation='sigmoid'))  # Layer output (untuk prediksi biner)
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model():
    X, y = generate_data()
    
    model = create_model(X.shape[1])
    
    print("[*] Melatih model...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)
  
    model.save('ml_model_V5.h5')
    print("[+] Model berhasil disimpan sebagai ml_model_V5.h5")

train_and_save_model()
