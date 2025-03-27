import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, Conv1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Optimize CPU Training kalau mau GPU tinggal hapus
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

# Load Dataset
DATASET_PATH = "dataset_quze.csv"
if not os.path.exists(DATASET_PATH):
    logging.error(f"❌ Dataset {DATASET_PATH} tidak ditemukan!")
    exit()

df = pd.read_csv(DATASET_PATH)
logging.info(f"📂 Dataset {DATASET_PATH} dimuat, total {len(df)} sampel")

# Preprocessing Data
X = np.array(df["payload"])
y = np.array(df["label"])

# Tokenisasi & Padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_WORDS = 200000  #  Kapasitas token lebih besar
MAX_LEN = 8192  #  Panjang payload lebih fleksibel

tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN)

def quantum_mutate(payload):
    noise_chars = ["%", "'", '"', "`", "(", ")", ";", "--", "/*", "*/", "+", "&", "|", "#", "\\", "0x", "[", "]"]
    for _ in range(random.randint(3, 15)):  
        idx = random.randint(0, len(payload) - 1)
        payload = payload[:idx] + random.choice(noise_chars) + payload[idx:]
    return payload

df["payload_mutated"] = df["payload"].apply(quantum_mutate)
X_mutated = np.array(df["payload_mutated"])
X_seq_mutated = tokenizer.texts_to_sequences(X_mutated)
X_pad_mutated = pad_sequences(X_seq_mutated, maxlen=MAX_LEN)

X_final = np.vstack((X_pad, X_pad_mutated))
y_final = np.hstack((y, y))

def quantum_noise_attack(payload_array, epsilon=0.2):
    noise = np.random.uniform(-epsilon, epsilon, payload_array.shape)
    return payload_array + noise

X_noisy = quantum_noise_attack(X_final, epsilon=0.2)
X_train = np.vstack((X_final, X_noisy))
y_train = np.hstack((y_final, y_final))
model = Sequential([
    Embedding(input_dim=MAX_WORDS, output_dim=768, input_length=MAX_LEN),
    Conv1D(filters=4096, kernel_size=7, activation='relu'),
    BatchNormalization(),
    Flatten(),
    Bidirectional(LSTM(8192, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)),
    Bidirectional(LSTM(4096, dropout=0.4, recurrent_dropout=0.4)),
    Dropout(0.3),
    Dense(4096, activation="relu"),
    Dense(2048, activation="relu"),
    Dense(1, activation="sigmoid")
])

class AIWAF_Evolver:
    def __init__(self):
        self.payload_memory = {}

    def update_feedback(self, payload, success):
        if payload not in self.payload_memory:
            self.payload_memory[payload] = {"success": 0, "fail": 0}
        if success:
            self.payload_memory[payload]["success"] += 1
        else:
            self.payload_memory[payload]["fail"] += 1

    def generate_best_payload(self, base_payload):
        mutations = ["%", "'", '"', "`", "(", ")", ";", "--", "/*", "*/", "0x", "[", "]"]
        for mutation in mutations:
            test_payload = base_payload.replace(" ", mutation)
            if test_payload in self.payload_memory and self.payload_memory[test_payload]["success"] > 20:
                return test_payload
        return base_payload

ai_waf_evolver = AIWAF_Evolver()
model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.000001), metrics=["accuracy"])
logging.info("✅ Model Mindfork AI WAF v6.1 berhasil dibuat!")
EPOCHS = 200
BATCH_SIZE = 2048
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)
logging.info("🎉 Training selesai")
MODEL_PATH = "ml_model_v6.h5"
model.save(MODEL_PATH)
logging.info(f"✅ Model berhasil disimpan di {MODEL_PATH}")
