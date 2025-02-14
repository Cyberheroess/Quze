import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Buat dataset dummy dengan 10 fitur
num_samples = 5000
num_features = 10

X = np.random.rand(num_samples, num_features)
y = np.random.randint(0, 2, size=(num_samples,))  # Klasifikasi biner

# Bagi dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Bangun model dengan lebih banyak lapisan dan neuron
model = Sequential([
    Dense(128, activation='relu', input_shape=(num_features,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')  # Output layer untuk klasifikasi biner
])

# Kompilasi model dengan optimizer yang lebih canggih dan adaptive learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Callback untuk menghentikan lebih awal dan mengurangi learning rate saat tidak ada peningkatan
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Latih model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), 
          callbacks=[early_stopping, reduce_lr])

# Simpan model
model.save("ml_model_v5.h5")
