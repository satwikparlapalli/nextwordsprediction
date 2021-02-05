from flask import Flask,jsonify, render_template                                                   
from flask import request
import json
import datetime
import random
import numpy as np
import pickle
import pandas as pd
import flask
#from flasgger import Swagger
import streamlit as st 
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import copy

from PIL import Image

app=Flask(__name__)


infile = open('tokenizers.pkl','rb')
tokenizer_encoder, tokenizer_decoder = pickle.load(infile)
infile.close()

def custom_lossfunction(targets,logits):

  # Custom loss function that will not consider the loss for padded zeros.
  # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention#define_the_optimizer_and_the_loss_function
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  mask = tf.math.logical_not(tf.math.equal(targets, 0))
  loss_ = loss_object(targets, logits)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

tf.keras.losses.custom_loss = custom_lossfunction

#Load the model
model = tf.keras.models.load_model('final_attention',custom_objects={'custom_lossfunction':custom_lossfunction})

@app.route('/predict')
def predict():
  return render_template('my_form.html')


@app.route('/predict',methods=['POST'])
def submit():
  input_sentence = request.form['input_sent']
  if request.form.get('multiple_predict'):
    result = predict_final(input_sentence)
  else:
    result = predict_single(input_sentence)

  return render_template('my_form.html', prediction_text=input_sentence+" {}".format(result))
  
def predict_final(input_sentence):
  encoder_test_tokens = tokenizer_encoder.texts_to_sequences([input_sentence])
  padded_encoder_input = pad_sequences(encoder_test_tokens, maxlen=14, dtype='float32', padding='post')
  encoder = model.layers[2]
  encoder_op, enc_h, enc_c = encoder(padded_encoder_input)
  decoder = model.layers[4]
  index_of_start = np.array(tokenizer_decoder.word_index['<start>']).reshape(1,1).astype('float32')
  predicted_out,enc_h, enc_c,attention,context_vector = decoder.onestepdecoder(index_of_start,encoder_op, enc_h, enc_c)
  state_h, state_c = enc_h,enc_c
  states = (state_h, state_c)
  toppred = np.argsort(predicted_out[0])[-3:][::-1]
  probs = np.sort(predicted_out[0])[-3:][::-1]
  words = []
  for pred in toppred:
    word = [k for k in tokenizer_decoder.word_index if tokenizer_decoder.word_index[k]==pred][0]
    words.append(word)
  semi_final = [[probs[0],[toppred[0]],[words[0]],states],[probs[1],[toppred[1]],[words[1]],states],[probs[2],[toppred[2]],[words[2]],states]]
  finished_sentences = 0
  final = []
  while (True):
    temp = []
    for i in range(len(semi_final)):
      # dec_emb= decoder.embedding(semi_final[i][1][-1].reshape(1,1))
      predicted_out,state_h, state_c,attention,context_vector = decoder.onestepdecoder(semi_final[i][1][-1].reshape(1,1).astype('float32'),
                                                                                       encoder_op, semi_final[i][-1][0], semi_final[i][-1][1])
      toppred = np.argsort(predicted_out[0])[-len(semi_final):][::-1]
      probs = np.sort(predicted_out[0])[-len(semi_final):][::-1]
      states= (state_h, state_c)
      for j in range(len(toppred)):
        word = [k for k in tokenizer_decoder.word_index if tokenizer_decoder.word_index[k]==toppred[j]][0]
        #temp[str(i)+','+str(j)] = (semi_final[i][0] * probs[j],toppred[j],semi_final[i][2].append(word),states)
        words = copy.deepcopy(semi_final[i][2])
        words.append(word)
        temp.append([semi_final[i][0] * probs[j],[toppred[j]],words,states])
    temp = sorted(temp,key = lambda x:x[0],reverse=True)[:len(semi_final)]
    ids_to_be_removed = []
    for id,k in enumerate(temp):
      if k[2][-1] == '<end>':
        final.append((k[0],' '.join(k[2][:-1])))
        finished_sentences+=1
        ids_to_be_removed.append(id)
    for id in ids_to_be_removed:
      temp[id] = 0
    temp = [i for i in temp if i!=0]
    semi_final=temp
    if finished_sentences == 3:
      break
  predictions_3 = [x[1] for x in final]

  return predictions_3
#jsonify(predictions_3) 


def predict_single(input_sentence):
  encoder_test_tokens = tokenizer_encoder.texts_to_sequences([input_sentence])
  padded_encoder_input = pad_sequences(encoder_test_tokens, maxlen=14, dtype='float32', padding='post')
  encoder = model.layers[2]
  encoder_op, enc_h, enc_c = encoder(padded_encoder_input)
  decoder = model.layers[4]
  index_of_start = np.array(tokenizer_decoder.word_index['<start>']).reshape(1,1).astype('float32')
  pred=0
  sentence = []
  attention_weights=[]
  att_wgts = tf.TensorArray(dtype=tf.float32, dynamic_size=True,size=0)
  while pred!=tokenizer_decoder.word_index['<end>']:
    predicted_out,enc_h, enc_c,attention,context_vector = decoder.onestepdecoder(index_of_start,encoder_op, enc_h, enc_c)
    pred = np.argmax(predicted_out) 
    word = [k for k in tokenizer_decoder.word_index if tokenizer_decoder.word_index[k]==(pred)][0]
    sentence.append(word)
    index_of_start = np.array(pred).reshape(1,1).astype('float32')

  return ' '.join(sentence[:-1])

if __name__ == '__main__':
    app.run(debug=True)
