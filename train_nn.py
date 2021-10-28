import re
import sys
import json
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,balanced_accuracy_score
import numpy as np

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

def getSequences(dataset,indexes):
    """
    Processamento do conjunto de dados para o formato onde cada linha contem uma palavra e uma classificação para 
    a palavra.
    OBS.: Função específica para lidar com a formatação do arquivo JSON.
    EX.:

        PALAVRA CLASSE
        PALAVRA CLASSE

    Para cada texto de tamannho N, retorna duas strings, sendo a primeira strings o próprio texto 
    e a segunda string também de tamanho N com uma classificação para cada palavra do texto.
    EX.: Considerando as classes BEM e COR

        O carro é azul e a casa é vermelha
        O BEM O COR O O BEM O COR
    """

    x_seqs = []
    y_seqs = []
    for i in indexes:
        text = dataset['text'][i]
        split_word = text.split(' ')
        y_seq = ['O' for i in range(len(split_word)+1)]
        entities = dataset['entities'][i]
        for ent in entities:
            currentTagWords = text[ent[0]: ent[1]].split()
            numWords = len(currentTagWords)
            countWord = 0
            countLen = 0
            countNum = 0
            countWordSinceStart = 0

        
            tempsplit = text[:ent[0]].split()
            countWordSinceStart = len(tempsplit)

            for p in range(len(split_word)):
                if p == countWordSinceStart:
                    y_seq[p] = f'{ent[2]}'
                    for j in range(1, numWords):
                        y_seq[p+j] = f'{ent[2]}'
        y_seqs.append(y_seq)
        x_seqs.append(dataset['text'][i])
    y_seqs = [" ".join([y_seq[i] for i in range(len(y_seq)) ]) for y_seq in y_seqs]
    return x_seqs,y_seqs

def zero_padded_f1(y_true, y_pred):
    """Métricas para avaliação do modelo de rede neural"""
    y_pred_flat, y_true_flat = [], []
    for y_pred_i, y_true_i in zip(y_pred.flatten(), y_true.flatten()):
        if y_true_i != 0:
            y_pred_flat.append(y_pred_i)
            y_true_flat.append(y_true_i)
    f1_macro = f1_score(y_true_flat, y_pred_flat, average='macro')
    precision = precision_score(y_true_flat, y_pred_flat, average='macro')
    recall = recall_score(y_true_flat, y_pred_flat, average='macro')
    acc = balanced_accuracy_score(y_true_flat, y_pred_flat)
    return f1_macro,precision,recall,acc

if __name__ == '__main__':
    # Carregando rotulações
    dataset = pd.read_json('textos_rotulados.json',lines=True)

    data_index = [i for i in dataset.index.values]
    print('Conjunto de dados com ',len(data_index),' textos')

    # Divisão simples em treino e teste
    train_indexes, test_indexes = train_test_split(
        data_index, test_size=TEST_SPLIT, random_state=RANDOM_STATE
    )

    # Tokenizando sequências de rótulos
    x_train,y_train = getSequences(dataset,train_indexes)
    x_test,y_test = getSequences(dataset,test_indexes)

    # Criação do tokenizador de classes
    label_tokenizer = Tokenizer(filters='', lower=False)
    label_tokenizer.fit_on_texts(y_train)

    # Como as classificações antes eram uma string, transformamos cada classificação de texto em uma sequência
    # resultando em um array de sequências de tokens
    y_train = label_tokenizer.texts_to_sequences(y_train)
    y_test = label_tokenizer.texts_to_sequences(y_test)

    word_tokenizer = Tokenizer(filters='',lower=True, oov_token='OOV')
    word_tokenizer.fit_on_texts(x_train)

    # Transformando os textos em sequências de tokens
    x_train = word_tokenizer.texts_to_sequences(x_train)
    x_test = word_tokenizer.texts_to_sequences(x_test)

    # Transformando as sequências em tamanho único (MAX_SEQ_LEN), para sequências menores completa com PAD.
    X_train = pad_sequences(maxlen=MAX_SEQ_LEN, sequences=x_train, padding='post')
    X_test = pad_sequences(maxlen=MAX_SEQ_LEN, sequences=x_test, padding='post')

    y_train = pad_sequences(maxlen=MAX_SEQ_LEN, sequences=y_train, padding='post')
    y_test = pad_sequences(maxlen=MAX_SEQ_LEN, sequences=y_test, padding='post')

    # One-hot-embedding
    Y_train = to_categorical(y_train,num_classes=len(label_tokenizer.word_index)+1)
    Y_test = to_categorical(y_test,num_classes=len(label_tokenizer.word_index)+1)

    vocab_size = len(word_tokenizer.word_index)+1

    print('Formato das sequências: ',X_train.shape)
    print('Tamanho do vocabulário: ', vocab_size)

    model = create_model(word_tokenizer,label_tokenizer)


    # Salvando somente os melhores modelos treinados
    weightspath = "NN/weights_bilstm.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(weightspath, monitor="val_accuracy",
        verbose=1, save_best_only=True, mode="max")
    earlyStopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', mode='min', patience=4
        )

    print(model.summary())
    print("Início do treinamento")
    model.fit(x=X_train,y=Y_train,validation_data=(X_test, Y_test),batch_size=BATCH_SIZE,verbose=1, epochs=500,shuffle=True, callbacks=[checkpoint,earlyStopping])


    # Obtendo métricas da rede
    nn_acc = []
    nn_f1 = []
    nn_precision = []
    nn_recall = []

    Y_pred_test, Y_pred_train = model.predict(X_test), model.predict(X_train)
    y_pred_test, y_pred_train = Y_pred_test.argmax(axis=2), Y_pred_train.argmax(axis=2)
    f1_macro,precision,recall,acc1 = zero_padded_f1(y_test,y_pred_test)

    nn_f1.append(f1_macro)
    nn_acc.append(acc1)
    nn_precision.append(precision)
    nn_recall.append(recall)

    log = {
            'nn_acc': nn_acc,
            'nn_f1': nn_f1,
            'nn_precision': nn_precision,
            'nn_recall': nn_recall,
        }

    # Salvando informações importantes

    with open('NN/log.json','w') as fp:
        json.dump(log,fp,indent=4)
    with open('NN/word_tokenizer.pkl', 'wb') as file_pi:
            pickle.dump(word_tokenizer, file_pi, protocol=pickle.HIGHEST_PROTOCOL)
    with open('NN/label_tokenizer.pkl', 'wb') as file_pi:
        pickle.dump(label_tokenizer, file_pi, protocol=pickle.HIGHEST_PROTOCOL)