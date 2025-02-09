import tensorflow as tf
import os
import subprocess

def check_tensorflow_cpu():
    try:
        # Mengecek apakah TensorFlow bisa menggunakan CPU
        with tf.device('/CPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
            c = tf.matmul(a, b)
        print("[✔] TensorFlow berjalan dengan CPU.")
        return True
    except Exception as e:
        print(f"[✘] TensorFlow tidak bisa berjalan di CPU: {e}")
        return False

def reinstall_tensorflow():
    print("[*] Menginstal ulang TensorFlow...")
    subprocess.run(["pip", "uninstall", "-y", "tensorflow"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    subprocess.run(["pip", "install", "tensorflow"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("[✔] Instalasi ulang selesai.")

def fix_tensorflow():
    if check_tensorflow_cpu():
        return
    print("[!] Mencoba memperbaiki TensorFlow...")

    # Coba perbaiki dengan setting ulang CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Coba cek ulang apakah TensorFlow bisa jalan setelah setting ulang
    if check_tensorflow_cpu():
        return

    # Jika masih gagal, coba instal ulang
    reinstall_tensorflow()

    # Cek lagi setelah instal ulang
    if check_tensorflow_cpu():
        print("[✔] TensorFlow berhasil diperbaiki!")
    else:
        print("[✘] TensorFlow masih bermasalah, coba install ulang secara manual.")

if __name__ == "__main__":
    fix_tensorflow()
