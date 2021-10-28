# sequence-labeling-bi-lstm
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

  Write a new file called "texts_predicted.txt" in root directory.
  
  
Code comments in portuguese.
