import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import re
import sys
import json
import pandas as pd

# Hiperparametros
EMB_OUT_DIM = 300
BATCH_SIZE = 32
LSTM_UNITS = 8
DENSE_UNITS = 8
MAX_SEQ_LEN = 25
DROPOUT_RATE = 0.2
TEST_SPLIT = 0.25
RANDOM_STATE = 42

def create_model(word_tokenizer,label_tokenizer):
    """
    Cria um modelo de rede neural Bi-LSTM utilizando Keras.
    """
    model_in = keras.layers.Input(shape=(MAX_SEQ_LEN,))
    model_hidden = keras.layers.Embedding(input_dim=len(word_tokenizer.word_index) + 2, 
            output_dim=EMB_OUT_DIM,input_length=MAX_SEQ_LEN, name="Embbeding_Layer", mask_zero=True)(model_in)
    model_hidden = keras.layers.Bidirectional(
            keras.layers.LSTM(units=LSTM_UNITS, return_sequences=True),
            name="BiLSTM_Layer"
        )(model_hidden)
    if DENSE_UNITS:
        model_hidden = keras.layers.TimeDistributed(keras.layers.Dense(
                    units=DENSE_UNITS, activation="relu"), name="Dense_Hidden_Layer")(model_hidden)
    model_out = keras.layers.TimeDistributed(keras.layers.Dense(units=len(label_tokenizer.word_index) + 1,activation="softmax"), name="Dense_Output_Layer")(model_hidden)

    model = keras.models.Model(model_in, model_out)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


if __name__ == '__main__':

    # Arquivo contendo as descrições a serem classificadas pela rede neural
    predict_texts_file = 'texts_to_predict.txt'
    with open(predict_texts_file,'r') as file:
        lines = file.readlines()

    # Lendo os tokenizadores obtidos durante o treinamento
    with open('NN/word_tokenizer.pkl', "rb") as handle:
        word_tokenizer = pickle.load(handle)

    with open('NN/label_tokenizer.pkl', "rb") as handle:
        label_tokenizer = pickle.load(handle)

    model = create_model(word_tokenizer,label_tokenizer)

    model.load_weights('NN/weights_bilstm.hdf5')

    # IDX -> TAG
    label_detokenizer = {v: k for k, v in label_tokenizer.word_index.items()}

    x_pred = [re.sub('\n','',line) for line in lines]
    x_pred = word_tokenizer.texts_to_sequences(x_pred)
    X_pred = pad_sequences(maxlen=MAX_SEQ_LEN, sequences=x_pred, padding='post')
    Y_pred = model.predict(X_pred)
    Y_pred = Y_pred.argmax(axis=2)


    # O resultados serão salvos nesse arquivo txt
    result_file = open('texts_predicted.txt','w')
    limited_seqs = []
    for i in range(len(Y_pred)):
        seq_len = len(x_pred[i])
        limited_seqs.append(Y_pred[i][:seq_len])

    for seq in limited_seqs:
        result = " ".join([label_detokenizer[l] for l in seq])
        result_file.write(result+'\n')

    result_file.close()