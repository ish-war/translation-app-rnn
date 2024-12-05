import keras
import collections
import numpy as np 
import json

from keras_preprocessing.text import Tokenizer
from keras_core.utils import pad_sequences
from keras_core.models import Model,Sequential
from keras_core.layers import Input,Dense,Embedding,GRU,LSTM,Bidirectional,Dropout,TimeDistributed,RepeatVector
from keras_core.optimizers import Adam
from keras_core.losses import sparse_categorical_crossentropy