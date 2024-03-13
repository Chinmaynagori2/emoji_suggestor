# app.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

from tensorflow.keras.saving import load_model


# x_tokens = tokenizer.texts_to_sequences(X)

# Necessary text preprocessing
necessary = [".", ",", "!", "?"]
def r_unnecessary1(line):
    nline = ""
    line = line.split()
#     line = ['sejbsgr','sejgbsr','akbrgogir']
#     words = 'sejbsgr'
    for words in line:
        c = 0
        if(words[0] in necessary):
            c = 1
        elif(words[(len(words)-1)] in necessary):
            c = 2
        if(c == 1):
            nline += words[0] + " " + words[1:len(words)] + " "
        if(c == 2):
            nline += words[0:(len(words) - 1)] + " " + words[(len(words) - 1)] + " "
        if(c == 0):
            nline += words + " "
    return nline

necessary = [".", ",", "!", "?"]
def r_unnecessary2(line):
    nline = ""
    line = line.split()
    for words in line:
        if((words.isalpha() == False) and (words in necessary)):
            nline += words + " "
        elif((words.isalpha() == False) and (words not in necessary)):
            continue
        else:
            words = words.lower()
            nline += words + " "
    return nline


# padding the sequence
from tensorflow.keras.utils import pad_sequences
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
def final_p(X):
    x_tokens = tokenizer.texts_to_sequences(X)
    padded = pad_sequences(
        x_tokens,
        maxlen=31,
        padding='post',
        truncating='pre',
        value=0.0
    )
    return padded

# Load the model
model = load_model('model/emoji_model5.h5')


emoji_mapping = {
    0: "😜",
    1: "📸",
    2: "😍",
    3: "😂",
    4: "😉",
    5: "🎄",
    6: "📷",
    7: "🔥",
    8: "😘",
    9: "❤️",
    10: "😁",
    11: "🇺🇸",
    12: "☀",
    13: "✨",
    14: "💙",
    15: "💕",
    16: "😎",
    17: "😊",
    18: "💜",
    19: "💯"
}

import numpy as np

# define the routes
@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    para = request.form['text']
    t1 = r_unnecessary1(para)
    t2 = r_unnecessary2(t1)
    t3 = final_p([t2])
    prediction = model.predict(t3)[0]
    prediction = np.argmax(prediction)
    emoji = emoji_mapping[prediction]

    return render_template('index.html', emoji_pred = "The emoji you should use is - " + emoji + "."
)

if __name__ == '__main__':
    app.run(debug=True)
