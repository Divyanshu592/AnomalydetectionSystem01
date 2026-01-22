import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lstm_autoencoder(window_size: int, n_features: int, latent_dim: int = 64):
    """
    Returns a TensorFlow LSTM Autoencoder model.
    input shape: (window_size, n_features)
    """
    inputs = layers.Input(shape=(window_size, n_features))

    # Encoder
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(latent_dim, return_sequences=False)(x)

    # Decoder
    x = layers.RepeatVector(window_size)(x)
    x = layers.LSTM(latent_dim, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(128, return_sequences=True)(x)

    outputs = layers.TimeDistributed(layers.Dense(n_features))(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse"
    )

    return model
