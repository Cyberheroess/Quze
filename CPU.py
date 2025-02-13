import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import os
import logging
import random

# **Konfigurasi TensorFlow**
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Matikan GPU jika perlu
print("Using TensorFlow version:", tf.__version__)

# **Variasi Nama Model Path (Sesuai Quze)**
model_paths = [
    "ml_model_v6.h5",
    "quze_ai_advanced_model.h5",
    "cyber_ai_v2.h5",
    "quantum_ml_advanced_model.h5",
    "adaptive_payload_model_v2.h5"
]

# **Logging untuk debugging**
logging.basicConfig(filename="training_log_advanced.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

# **1. Buat Dataset AI untuk Pengenalan Pola Mutasi Payload**
def generate_mutation_data(samples=5000):
    """
    Membuat dataset AI untuk mengenali pola mutasi payload SQLi, XSS, dan WAF bypass.
    """
    patterns = [
        "' OR 1=1 --", "<script>alert('XSS')</script>", "'; DROP TABLE users; --",
        "admin' --", "' UNION SELECT * FROM users --"
    ]

    variations = [
        "' OR 'a'='a' --", "<img src=x onerror=alert('XSS')>",
        "'; EXEC xp_cmdshell('dir'); --", "' or 1=1#",
        "' UNION SELECT username, password FROM users --"
    ]

    X, y = [], []
    for _ in range(samples):
        idx = random.randint(0, len(patterns) - 1)
        X.append([ord(c) for c in patterns[idx].ljust(10)])  # Padding ke 10 karakter
        y.append([ord(c) for c in variations[idx].ljust(10)])  # Padding ke 10 karakter

    return np.array(X), np.array(y)

X_train, y_train = generate_mutation_data(5000)
X_test, y_test = generate_mutation_data(1000)

# **2. Bangun Model AI untuk Mutasi Payload (Dengan LSTM dan Regularisasi)**
def create_mutation_model():
    inputs = Input(shape=(10,))
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(inputs)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='linear')(x)  # Output 10 karakter termutasi
    
    model = Model(inputs, x)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=['accuracy'])
    return model

model = create_mutation_model()
model.summary()

# **3. Latih Model AI dengan Learning Rate Scheduler dan Callback**
def train_model(model, X_train, y_train, X_test, y_test):
    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    print("[*] Training AI model untuk mutasi payload...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

    # **4. Validasi Model Sebelum Disimpan**
    loss, accuracy = model.evaluate(X_test, y_test)
    if accuracy > 0.80:  # Target akurasi lebih tinggi
        for path in model_paths:
            model.save(path)
            logging.info(f"[+] Model disimpan sebagai {path} dengan akurasi {accuracy:.2f}")
            print(f"[+] Model disimpan sebagai {path} dengan akurasi {accuracy:.2f}")
    else:
        logging.error(f"[-] Model tidak memenuhi syarat, akurasi hanya {accuracy:.2f}")
        print(f"[-] Model tidak memenuhi syarat, akurasi hanya {accuracy:.2f}. Latih ulang diperlukan.")

train_model(model, X_train, y_train, X_test, y_test)
