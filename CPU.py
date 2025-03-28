import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, Input, Conv1D, Flatten, BatchNormalization
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
import os
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Optimasi CPU Multithreading
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.set_soft_device_placement(True)

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

MAX_WORDS = 500000  # Upgrade kapasitas token
MAX_LEN = 16384  # Payload lebih panjang

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<UNK>")
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=MAX_LEN)

# **Adversarial Payload Mutation (GAN-Based)**
def adversarial_mutate(payload):
    mutation_rules = ["%", "'", '"', "`", "(", ")", ";", "--", "/*", "*/", "+", "&", "|", "#", "\\", "0x", "[", "]"]
    for _ in range(random.randint(10, 30)):  # Lebih banyak mutasi adaptif
        idx = random.randint(0, len(payload) - 1)
        payload = payload[:idx] + random.choice(mutation_rules) + payload[idx:]
    return payload

df["payload_mutated"] = df["payload"].apply(adversarial_mutate)
X_mutated = np.array(df["payload_mutated"])
X_seq_mutated = tokenizer.texts_to_sequences(X_mutated)
X_pad_mutated = pad_sequences(X_seq_mutated, maxlen=MAX_LEN)

X_final = np.vstack((X_pad, X_pad_mutated))
y_final = np.hstack((y, y))

# **Gaussian Noise Augmentation**
def gaussian_noise(payload_array, std_dev=0.1):  # Optimasi noise agar tetap stealthy
    noise = np.random.normal(0, std_dev, payload_array.shape)
    return payload_array + noise

X_noisy = gaussian_noise(X_final, std_dev=0.08)
X_train = np.vstack((X_final, X_noisy))
y_train = np.hstack((y_final, y_final))

# **Advanced Transformer Model**
def build_advanced_transformer():
    inputs = Input(shape=(MAX_LEN,))
    x = Embedding(input_dim=MAX_WORDS, output_dim=1024, input_length=MAX_LEN)(inputs)
    x = Conv1D(filters=2048, kernel_size=5, activation='relu')(x)
    x = BatchNormalization()(x)

    # **Upgraded Multi-Head Attention**
    attention = MultiHeadAttention(num_heads=16, key_dim=128)(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attention)

    x = Flatten()(x)
    x = Dense(4096, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    return Model(inputs, outputs)

model = build_advanced_transformer()

# **Self-Learning AI (Reinforcement Learning)**
class AIWAF_Reinforcement:
    def __init__(self):
        self.payload_memory = {}

    def update_feedback(self, payload, success):
        if payload not in self.payload_memory:
            self.payload_memory[payload] = {"success": 0, "fail": 0}
        if success:
            self.payload_memory[payload]["success"] += 1
        else:
            self.payload_memory[payload]["fail"] += 1

    def evolve_payload(self, base_payload):
        mutations = ["%", "'", '"', "`", "(", ")", ";", "--", "/*", "*/", "0x", "[", "]"]
        best_payload = base_payload

        for mutation in mutations:
            test_payload = base_payload.replace(" ", mutation)
            if test_payload in self.payload_memory and self.payload_memory[test_payload]["success"] > 50:  # Lebih ketat dalam validasi payload sukses
                best_payload = test_payload

        return best_payload

ai_waf_rl = AIWAF_Reinforcement()

# **Adaptive Monte Carlo Attack Strategy**
class MonteCarloPlanner:
    def __init__(self, exploration_rate=0.3):
        self.exploration_rate = exploration_rate
        self.payload_tree = {}

    def select_best_payload(self, base_payload):
        if random.random() < self.exploration_rate:  
            return adversarial_mutate(base_payload)  
        return ai_waf_rl.evolve_payload(base_payload)  

attack_planner = MonteCarloPlanner()

# **Optimasi Learning Rate dengan Cosine Decay**
initial_learning_rate = 0.00005
lr_schedule = CosineDecay(initial_learning_rate, decay_steps=25000, alpha=0.00001)
optimizer = Adam(learning_rate=lr_schedule)

model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
logging.info("✅ Model Quzee AI WAF v7.1 berhasil di-upgrade!")

EPOCHS = 350  # Ditambah untuk adaptasi lebih baik
BATCH_SIZE = 8192  # Upgrade ke batch besar biar lebih stabil
history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

logging.info("🎉 Training selesai")
MODEL_PATH = "ml_model_v7.1.h5"
model.save(MODEL_PATH)
logging.info(f"✅ Model berhasil disimpan di {MODEL_PATH}")
