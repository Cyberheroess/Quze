import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[+] GPU Terdeteksi: {gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    print("[!] Tidak ada GPU, menggunakan CPU...")
    
def generate_data():
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=(10000, 10))  # **10 fitur sesuai kebutuhan Quze**
    y = np.random.randint(0, 2, 10000)  # Label biner (0 atau 1)
    return X, y

def create_model(input_shape):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # **Output biner karena Quze pakai binary classification**
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def train_and_save_models():
    X, y = generate_data()
    model = create_model(X.shape[1])

    print("[*] Mulai pelatihan model...")

    checkpoint_path = "best_model_v5.keras"

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    model.fit(X, y, epochs=50, batch_size=64, validation_split=0.2, callbacks=callbacks)

    print("[*] Validasi model dengan input test...")
    test_input = np.random.rand(1, 10) 
    try:
        test_output = model.predict(test_input)
        print(f"[+] Model valid, contoh output: {test_output}")
    except Exception as e:
        print(f"[!] ERROR: Model gagal dipakai untuk inferensi: {e}")
        return

    filenames = [
        "ml_model_V5.h5",
        "ml_model_v5.h5",
        "ml_Model_V5.h5",
        "ML_MODEL_V5.h5",
        "ml_model_v5.keras",
        "ml_model_v5_backup.h5"
    ]

    for name in filenames:
        if name.endswith(".h5"):
            model.save(name, save_format="h5")
        else:
            model.save(name)  # Keras format default
        print(f"[+] Model berhasil disimpan sebagai {name}")

train_and_save_models()
