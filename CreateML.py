import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import numpy as np
import datetime

# **Cek apakah GPU tersedia & gunakan jika ada**
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[+] GPU Ditemukan: {gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("[!] Tidak ada GPU, menggunakan CPU...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# **Fungsi untuk membuat dataset yang lebih realistis**
def generate_data():
    """Membuat dataset dengan distribusi lebih baik dan jumlah lebih besar."""
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(10000, 15))  # Tambah dimensi fitur
    y = np.random.randint(0, 2, 10000)  # Label biner
    return X, y

# **Fungsi untuk membuat model AI yang lebih canggih**
def create_model(input_shape):
    """Membangun model dengan Batch Normalization untuk stabilitas & Dropout optimal."""
    model = Sequential([
        Dense(256, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        BatchNormalization(),
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

# **Fungsi utama untuk melatih dan menyimpan model**
def train_and_save_models():
    X, y = generate_data()
    model = create_model(X.shape[1])

    print("[*] Mulai pelatihan model...")

    # **Callback untuk optimasi training**
    checkpoint_path = "best_model_v5.h5"
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=log_dir)
    ]

    # **Latih model dengan callback**
    model.fit(X, y, epochs=100, batch_size=64, validation_split=0.2, callbacks=callbacks)

    # **Simpan model dalam berbagai format**
    filenames = [
        "ml_model_V5.h5",
        "ml_model_v5.keras",
        "ml_MODEL_V5.keras",
        "ml_model_v5_backup.h5"
    ]
    for name in filenames:
        if name.endswith(".h5"):
            model.save(name, save_format="h5")
        else:
            model.save(name)  # Keras format default
        print(f"[+] Model berhasil disimpan sebagai {name}")

train_and_save_models()
