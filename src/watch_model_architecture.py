from tensorflow.python.keras.backend import log
from utils import log_model_architecture

log_model_architecture(model_path="src/model/trained_model/discriminator_example.h5")
log_model_architecture(model_path="src/model/trained_model/generator_example.h5")