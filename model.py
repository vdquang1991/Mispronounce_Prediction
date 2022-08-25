import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Dense, Dropout, GlobalAvgPool1D
from tensorflow.keras.layers import ZeroPadding1D, LSTM, Bidirectional, Lambda, Flatten


def cnn_model(input_shape = (32000,1), output=3, drop_rate=0.5):
    inp = Input(shape=input_shape)
    x = Conv1D(16, kernel_size=11, strides=8, padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(16, kernel_size=7, strides=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(32, kernel_size=7, strides=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = ZeroPadding1D(padding=3)(x)

    x = Conv1D(32, kernel_size=7, strides=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv1D(128, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Flatten()(x)
    x = Dropout(drop_rate)(x)
    out = Dense(output, activation='softmax')(x)
    return Model(inputs=inp, outputs=out)

def segment_encoded_signal(x, num_filters_in_encoder=64, chunk_size=256):
    num_full_chunks = 16000 * 2 // chunk_size
    signal_length_samples = chunk_size * num_full_chunks
    chunk_advance = chunk_size // 2
    x1 = tf.reshape(x, (-1, signal_length_samples//chunk_size, chunk_size, num_filters_in_encoder))
    x2 = tf.roll(x, shift=-chunk_advance, axis=1)
    x2 = tf.reshape(x2, (-1, signal_length_samples//chunk_size, chunk_size, num_filters_in_encoder))
    x2 = x2[:, :-1, :, :] # Discard last segment with invalid data

    x_concat = tf.concat([x1, x2], axis=1)
    x = x_concat[:, ::num_full_chunks, :, :]
    for i in range(1, num_full_chunks):
        x = tf.concat([x, x_concat[:, i::num_full_chunks, :, :]], axis=1)
    return x

def lstm_model(input_shape=(32000,1), output=3, drop_rate=0.5):
    inp = Input(shape=input_shape)
    x = Conv1D(32, kernel_size=11, strides=4, padding='same')(inp)
    x = Bidirectional(LSTM(16, return_sequences=True))(x)
    x = Conv1D(64, kernel_size=7, strides=4, padding='same')(x)
    x = Bidirectional(LSTM(32, return_sequences=True))(x)
    x = Conv1D(128, kernel_size=7, strides=4, padding='same')(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Conv1D(256, kernel_size=7, strides=4, padding='same')(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Conv1D(512, kernel_size=7, strides=4, padding='same')(x)
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Conv1D(512, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAvgPool1D()(x)
    x = Dropout(drop_rate)(x)
    out = Dense(output, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)



# model = cnn_model()
# model.summary(line_length=150)

