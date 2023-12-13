import os
import numpy as np
import pandas as pd
import re
import tensorflow
from LSTM import *
#
# word_index, index_word, word_vector_map, word_vector = load_dicts_from_json("word_dicts.json")
# index_output, output_index, type_map = load_dicts_from_json("output_dicts.json")
# loaded_model = tensorflow.keras.models.load_model('my_model.keras')
#
# print(type_map)
# play = np.array(["EMTRICITABINE VIREAD"])
# play_indices = sentences_to_indices(play, word_index, 8)
# # print(np.argmax(loaded_model.predict(play_indices), axis=1))
# #
# #
# # def map_value(value):
# #     return index_output.get(value, value)
# #
# # mapped_array = np.vectorize(map_value)(np.argmax(loaded_model.predict(play_indices), axis=1))
# # print(pd.Series(mapped_array))
# print(play[0] + ' ' + index_output[np.argmax(loaded_model.predict(play_indices))])
print(np.random.choice(2, p = [0.6,0.4]))