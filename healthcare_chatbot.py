# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:58:36 2023

@author: lboyd
"""



#imports the neccessary libraries to use within the project

import numpy
import tflearn
import tensorflow as tf 
from tensorflow.python.framework import ops 
import random
import pickle
import time 
import json
import tkinter as tk
import nltk
import tflearn
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
import pickle
import json
import random


strt = time.time()

stemmer = LancasterStemmer() #initializes a variable to use LancasterStemmer

with open(r"C:\Users\lboyd\OneDrive\Desktop\Year 3\Complex Systems\CW2\intents.json") as file:
    data = json.load(file) #opens and lodas the json file that contains the intents for the chatbot to use. 

with open("data.pickle", "rb") as f:
    words, labels, training, output = pickle.load(f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=300, batch_size=10, show_metric=True)
model.save("model.tflearn")

end = time.time()
total_time = (end - strt)
print ("Total time it takes to train model: ", total_time, "seconds")


model.load("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

def get_response(msg):
    results = model.predict([bag_of_words(msg, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
           

    return random.choice(responses)

class GUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Healthcare Chatbot")
        self.window.geometry("900x500")
        self.window.resizable(width=False, height=False)
        self.window.configure(bg="light blue")
        

        self.chatWindow = tk.Text(self.window, bd=1, bg="white", width="50", height="8",
                                  font=("Serif", 12), foreground="#0C0C0C")
        self.chatWindow.place(x=6, y=6, height=385, width=870)

        self.messageWindow = tk.Text(self.window, bd=0, bg="white", width="30", height="4",
                                     font=("Serif", 12), foreground="#0C0C0C")
        self.messageWindow.place(x=128, y=400, height=88, width=700)

        self.scrollbar = tk.Scrollbar(self.window, command=self.chatWindow.yview, cursor="star")
        self.scrollbar.place(x=880, y=5, height=385)

        self.button = tk.Button(self.window, text="Send", width="12", height=5,
                                 bd=0, bg="#828181", activebackground="#C1BEBE",
                                 foreground='#ffffff', font=("Serif", 12), command=self.send_message)
        self.button.place(x=6, y=400, height=88)

    def send_message(self):
        message = self.messageWindow.get("1.0", "end-1c").strip()
        self.messageWindow.delete("1.0", tk.END)
        self.chatWindow.config(state=tk.NORMAL)
        self.chatWindow.insert(tk.END, "You: " + message + "\n")

        response = get_response(message)
        self.chatWindow.insert(tk.END, "Chatbot: " + response + "\n")
        self.chatWindow.config(state=tk.DISABLED)
        self.chatWindow.see(tk.END)
        

    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    gui = GUI()
    gui.run
    
