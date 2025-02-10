import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import numpy as np
import datetime

# **Cek & Pakai GPU Jika Ada**
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[+] GPU Ditemukan: {gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Pakai GPU utama
    tf.keras.mixed_precision.set_global_policy("mixed_float16")  # Optimasi precision
else:
    print("[!] Tidak ada GPU, menggunakan CPU...")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# **Fungsi untuk membuat dataset besar dengan noise minimal**
def generate_data():
    """Membuat dataset besar dengan distribusi yang lebih kompleks."""
    np.random.seed(42)
    X = np.random.uniform(low=-1.0, high=1.0, size=(20000, 20))  # Tambah dimensi & distribusi lebih kompleks
    y = np.random.randint(0, 2, 20000)  # Label biner
    return X, y

# **Fungsi untuk membuat model AI yang lebih canggih & optimal untuk GPU**
def create_model(input_shape):
    """Membangun model dengan Batch Normalization, Dropout lebih optimal, dan arsitektur lebih dalam."""
    model = Sequential([
        Dense(512, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid', dtype=tf.float32)  # Pastikan output tetap float32 meski pakai mixed precision
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# **Fungsi utama untuk melatih dan menyimpan model dengan GPU**
def train_and_save_models():
    X, y = generate_data()
    model = create_model(X.shape[1])

    print("[*] Mulai pelatihan model di GPU...")

    # **Callback untuk optimasi training**
    checkpoint_path = "best_model_gpu.h5"
    log_dir = f"logs/fit/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        TensorBoard(log_dir=log_dir)
    ]

    # **Latih model dengan GPU**
    with tf.device('/GPU:0'):
        model.fit(X, y, epochs=150, batch_size=128, validation_split=0.2, callbacks=callbacks)

    # **Simpan model dalam berbagai format**
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
            model.save(name)  # Format default Keras
        print(f"[+] Model GPU berhasil disimpan sebagai {name}")

train_and_save_models()
