# GPU test for validating WSL linkage
import tensorflow as tf

# Check if GPU is detected
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[/] Success! Found GPU: {gpus[0]}")
    # This bit helps prevent TF from hogging all VRAM immediately
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("[X] GPU not found. Check your activation script paths.")