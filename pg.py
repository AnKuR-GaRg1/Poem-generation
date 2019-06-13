#importing all important packages

from tkinter import *
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import RNN
from keras.utils import np_utils
import random

#defining the action on clicking the button

def bt1():
    messagebox.showinfo("Information","please wait")

    #opening the text file
    
    text1= (open('sonnets.txt').read())
    text1=text1.lower()

    #getting list of all unique characters
    characters = sorted(list(set(text1)))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    X = []
    Y = []
    length = len(text1)
    seq_length = 400
    for i in range(0, length-seq_length, 1):
         sequence = text1[i:i + seq_length]
         label =text1[i + seq_length]
         X.append([char_to_n[char] for char in sequence])
         Y.append(char_to_n[label])


    X_modified = np.reshape(X, (len(X), seq_length, 1))
    X_modified = X_modified / float(len(characters))
    Y_modified = np_utils.to_categorical(Y)


    print("aaaaaaaaa")
    model = Sequential()
    model.add(LSTM(400, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(400))
    model.add(Dropout(0.2))
    model.add(Dense(Y_modified.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam' )
    model.load_weights('text_generator_400_0.2_400_0.2_400_0.2_100.h5')


    string_mapped = X[random.randint(0,10000)]
    full_string = [n_to_char[value] for value in string_mapped]
    # generating characters
    for i in range(400):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        full_string.append(n_to_char[pred_index])

        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]



        #combining text
    txt=" "
    for char in full_string:
        txt = txt+char
    print(txt)
    root3=Tk()
    frame1=Frame(root3,width=500,height=500)
    frame1.pack()
    poem=Label(root3,text=txt)
    poem.place(x=50,y=50)
        


root=Tk()
frame = Frame(root,width=200, height=100)
frame.pack()
root.title("poem generation")
button1=Button(root,text="generate",command=bt1,width=50, height=30,activeforeground='blue',activebackground='black')
button1.pack(side=TOP,padx=10,anchor="c")

root.mainloop()
