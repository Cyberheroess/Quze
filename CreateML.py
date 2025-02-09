import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import numpy as np
import datetime

# Pastikan TensorFlow hanya menggunakan CPU jika tidak ada GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

# Fungsi untuk membuat dataset
def generate_data():
    """Membuat dataset dengan distribusi lebih realistis."""
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(5000, 10))  # Data distribusi normal
    y = np.random.randint(0, 2, 5000)  # Label biner
    return X, y

# Fungsi untuk membuat model AI yang lebih optimal
def create_model(input_shape):
    """Membangun model AI dengan dropout untuk mencegah overfitting."""
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Fungsi utama untuk melatih dan menyimpan model
def train_and_save_models():
    X, y = generate_data()
    model = create_model(X.shape[1])

    print("[*] Mulai pelatihan model...")

    # Callback untuk early stopping & penyimpanan terbaik
    checkpoint_path = "best_model.h5"
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        TensorBoard(log_dir=f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")
    ]

    # Latih model dengan callback
    model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2, callbacks=callbacks)

    # Simpan model dalam berbagai format
    filenames = [
    "ml_model_V5.h5",
    "ml_model_v5.h5",
    "ml_Model_V5.h5",
    "ML_MODEL_V5.h5",
    "ml_model_v5.keras",
    "ml_MODEL_V5.keras",
    "ML_Model_v5.keras",
    "ml_model_v5_backup.h5"
]
    for name in filenames:
        if name.endswith(".h5"):
            model.save(name, save_format="h5")
        else:
            model.save(name)  # Keras format default
        print(f"[+] Model berhasil disimpan sebagai {name}")

train_and_save_models()
