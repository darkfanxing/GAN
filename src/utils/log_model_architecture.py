from tensorflow.keras.models import load_model

def log_model_architecture(model_path):
    model = load_model(model_path)
    print(model.summary())