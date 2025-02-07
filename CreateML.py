import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

def generate_data():
    X = np.random.rand(1000, 10)  # Data acak
    y = np.random.randint(0, 2, 1000)  # Label biner
    return X, y

def create_model(input_shape):
    model = Sequential([
        Dense(64, input_shape=(input_shape,), activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_models():
    X, y = generate_data()
    model = create_model(X.shape[1])

    print("[*] Melatih model...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    # Simpan dengan berbagai variasi nama
    filenames = [
        "ml_model_V5.h5",
        "ml_model_v5.h5",
        "ml_Model_V5.h5",
        "ml_model_V5.H5",
        "ML_MODEL_V5.h5",
        "ml_model_v5.keras"
    ]

    for name in filenames:
        if name.endswith(".h5"):
            model.save(name, save_format="h5")
        else:
            model.save(name)  # Keras format default
        print(f"[+] Model berhasil disimpan sebagai {name}")

train_and_save_models()
