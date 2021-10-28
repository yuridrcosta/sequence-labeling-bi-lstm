# sequence-labeling-bi-lstm
[Clique aqui](https://github.com/yuridrcosta/sequence-labeling-bi-lstm#pt-br) para a explicação em português.

A simple Keras implementation of a BI-LSTM neural network for Sequence Labeling/Named Entity Recognition


## Training Input

A DataFrame with columns **text** and **entities**.

  - **Column text:** String
  - **Column entities**: A list of entities descriptions, including starting and ending character position (integer) in the text and entity name (string),  according to the following format: <br/>

```
  [  [START_POS, END_POS,"ENTITY_TYPE1"], [START_POS, END_POS,"ENTITY_TYPE2"] ]
```

  Dataset loading is made in **line 112** of train_nn.py file.
  
## Predict input

 A txt file with one text per line.
 Texts loading is made in **line 50** of predict_nn.py file.
 
## Predict result

  Write a new file in root directory named "texts_predicted.txt" with classifications for each word of given texts.
  
 
## Execution 

With needed files, first install dependencies using
````
pip install -r requirements.txt
````
After, start training 
````
 python3 train_nn.py
````
To predict, use this command
````
python3 predict_nn.py
````

Code comments in portuguese.


# PT-BR

O repositório contém uma implementação simples de uma rede neural BI-LSTM para Reconhecimento de Entidades Nomeadas/Sequence Labeling utilizando Keras.

## Entrada de treinamento

  A entrada é um DataFrame contendo as colunas **text** e **entities**:
  - **Coluna text**: String
  - **Coluna entities**: Uma lista de descrições de entidades, contendo a posição dos caracteres de início e fim (inteiros) dentro do texto e o nome da entidade (string), de acordo com o seguinte formato: <br/>

```
  [  [POS_INICIO, POS_FIM,"ENTIDADE1"], [POS_INICIO, POS_FIM,"ENTIDADE2"] ]
```

  O carregamento do conjunto de dados é feito na **linha 112** do arquivo train_nn.py.
  
## Entrada de predição

 Um arquivo txt contendo um texto por linha.
 O carregamento do arquivo txt é feito na **linha 50** do arquivo predict_nn.py.

## Resultados de predição

  Um arquivo txt é gerado no diretório raíz nomeado "text_predicted.txt" contendo as classificações para cada palavra dos textos fornecidos.
 
## Execução 

Com os arquivos necessários, primeiro instale as dependências
````
pip install -r requirements.txt
````
Em seguida, realize o treinamento do modelo de rede neural
````
 python3 train_nn.py
````
Para predizer, utilize o seguinte código
````
python3 predict_nn.py
````

  
