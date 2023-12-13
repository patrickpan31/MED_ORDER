import os
import numpy as np
import pandas as pd
import re
import tensorflow
from tensorflow import keras
import json




def load_data(path1, path2):

    df = pd.read_excel(path1, sheet_name=None)
    combined_df = pd.concat(df.values(), ignore_index=True)
    target_columns = combined_df[['DRUG_NAME', 'ART', 'TYPE']]
    df_no_duplicates = target_columns.drop_duplicates()

    new_art = pd.read_csv(path2)
    new_art.columns = ['DRUG_NAME', 'ART', 'TYPE']
    result = pd.concat([df_no_duplicates, new_art]).reset_index(drop=True)
    # df_no_duplicates_reset = df_no_duplicates.reset_index(drop=True)

    return result


def output_transform(df):

    output = df['ART'].str.upper().explode().unique().tolist()
    output_index = {word:index for index, word in enumerate(output)}
    index_output = {index:word for index, word in enumerate(output)}
    df['ART'] = df['ART'].map(output_index)
    y = df['ART'].values
    print(y)
    return y, len(output_index), index_output, output_index


def type_mapping(df):

    newdf = df[['ART', 'TYPE']]
    newdf = newdf.drop_duplicates().reset_index(drop = True)

    type_map = {row.ART: row.TYPE for row in newdf.itertuples(index=False)}
    type_map['JULUCA'] = 'FDC2'
    type_map['DOVATO'] = 'FDC2'

    return type_map

def split_data(x,y):

    test_size = int(0 * len(x))

    # Shuffle the data
    indices = np.arange(len(x))
    np.random.shuffle(indices)
    X = x[indices]
    y = y[indices]

    X_train = X[test_size:]
    y_train = y[test_size:]
    X_test = X[:test_size]
    y_test = y[:test_size]

    return X_train, y_train, X_test, y_test


def word_dic_generator(df):

    unique_word = df['DRUG_NAME'].str.upper().str.split().explode().str.split(r'[,/-]').explode().unique().tolist()

    word_index = {word: index+1 for index, word in enumerate(unique_word)}
    index_word = {index+1: word for index, word in enumerate(unique_word)}

    word_vector = np.zeros((len(unique_word)+1, len(unique_word)))
    word_vector_map = {}

    for index, word in enumerate(unique_word):
        word_vector[index+1,index] = 1
        word_vector_map[index+1] = word_vector[index+1]
    print(word_index['EFAV'])

    return word_index, index_word, word_vector_map, word_vector


def save_word_dicts_as_json(word_index, index_word, word_vector_map, word_vector, filename):

    word_vector = word_vector.tolist()
    word_vector_map = {str(key): value.tolist() for key, value in word_vector_map.items()}

    data = {
        'word_index': word_index,
        'index_word': index_word,
        'word_vector_map': word_vector_map,
        'word_vector': word_vector
    }

    with open(filename, 'w') as file:
        json.dump(data, file)


def save_output_word_dicts_as_json(index_output, output_index, type_map, filename):
    data = {
        'index_output': index_output,
        'output_index': output_index,
        'type_map': type_map
    }

    with open(filename, 'w') as file:
        json.dump(data, file)


def load_dicts_from_json(filename):

    with open(filename, 'r') as file:
        data = json.load(file)

    if 'word_index' in data:

        index_word = {int(key): value for key, value in data['index_word'].items()}
        word_vector_map = {int(key): np.array(value) for key, value in data['word_vector_map'].items()}
        word_vector = np.array(data['word_vector'])

        return data['word_index'], index_word, word_vector_map, word_vector

    else:

        index_output = {int(key): value for key, value in data['index_output'].items()}
        return index_output, data['output_index'], data['type_map']


def sentences_to_indices(df, word_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()`.

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_index -- a dictionary containing each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    if isinstance(df, pd.DataFrame):
        m = df['DRUG_NAME'].shape[0]  # number of training examples
        x_indices = np.zeros((m, max_len), dtype=int)
        print(m)

        for i, row in df.iterrows():
            words = re.sub(r'[-,/]+', ' ', row['DRUG_NAME'].upper())
            words = [word.strip() for word in words.split(' ') if word.strip()][:max_len]
            x_indices[i, :len(words)] = [word_index.get(w, 0) for w in words]
        print(x_indices)

    else:
        m = df.shape[0]
        x_indices = np.zeros((m, max_len), dtype=int)
        for index, value in enumerate(df):
            words = re.sub(r'[-,/]+', ' ', value.upper())
            words = [word.strip() for word in words.split(' ') if word.strip()][:max_len]
            x_indices[index, :len(words)] = [word_index.get(w, 0) for w in words]

    return x_indices


def pretrained_embedding_layer(word_vector_map, word_vector):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_size = len(word_vector)
    # any_word = list(word_vector_map.keys())[0]
    emb_dim = word_vector_map[1].shape[0]

    print(vocab_size, emb_dim)

    # Define Keras embedding layer with the correct input and output sizes
    # Make it non-trainable.
    embedding_layer = tensorflow.keras.layers.Embedding(vocab_size, emb_dim, trainable=False)

    # Build the embedding layer, it is required before setting the weights of the embedding layer.
    embedding_layer.build((None,))  # Do not modify the "None".  This line of code is complete as-is.

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([word_vector])

    return embedding_layer


def Emojify_V2(input_shape, word_vector_map, word_vector, output_shape):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """

    # Define sentence_indices as the input of the graph.
    # It should be of shape input_shape and dtype 'int32' (as it contains indices, which are integers).
    sentence_indices = tensorflow.keras.Input(shape=input_shape, dtype='int32')

    # Create the embedding layer pretrained with GloVe Vectors (≈1 line)
    embedding_layer = pretrained_embedding_layer(word_vector_map, word_vector)

    # Propagate sentence_indices through your embedding layer
    # (See additional hints in the instructions).
    embeddings = embedding_layer(sentence_indices)

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # The returned output should be a batch of sequences, So, set return_sequences = True
    # If return_sequences = False, the LSTM returns only tht last output in output sequence
    X = tensorflow.keras.layers.LSTM(units=128, return_sequences=True)(embeddings)
    # Add dropout with a probability of 0.5
    X = tensorflow.keras.layers.Dropout(0.5)(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # The returned output should be a single hidden state, not a batch of sequences.
    X = tensorflow.keras.layers.LSTM(units=64, return_sequences=False)(X)
    # Add dropout with a probability of 0.5
    X = tensorflow.keras.layers.Dropout(0.5)(X)
    # Propagate X through a Dense layer with y units
    X = tensorflow.keras.layers.Dense(output_shape)(X)
    # Add a softmax activation
    X = tensorflow.keras.layers.Activation('softmax')(X)

    # Create Model instance which converts sentence_indices into X.
    model = tensorflow.keras.Model(inputs=sentence_indices, outputs=X)

    ### END CODE HERE ###

    return model

if __name__ == '__main__':

    df = load_data('ART_MED_1109.xlsx', 'new_art.csv')
    type_map = type_mapping(df)
    word_index, index_word, word_vector_map, word_vector = word_dic_generator(df)
    # sentences_to_indices(df, word_index, 8)
    pretrained_embedding_layer(word_vector_map, word_vector)

    features = sentences_to_indices(df, word_index, 8)
    labels, length, index_output, output_index = output_transform(df)
    print(len(features), len(labels))

    X_train, y_train, X_test, y_test = split_data(features, labels)

    Y_train_oh = tensorflow.one_hot(y_train, depth=length)

    model = Emojify_V2((8,), word_vector_map, word_vector, length)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, Y_train_oh, epochs=800, batch_size=32, shuffle=True)

    # Y_test_oh = tensorflow.one_hot(y_test, depth=length)
    # loss, acc = model.evaluate(X_test, Y_test_oh)
    # print()
    # print("Test accuracy = ", acc)

    play = np.array(["ELVITEGRAVIR"])
    play_indices = sentences_to_indices(play, word_index, 8)
    print(play[0] + ' ' + index_output[np.argmax(model.predict(play_indices))])

    model.save('my_model.keras')

    save_word_dicts_as_json(word_index, index_word, word_vector_map, word_vector, 'word_dicts.json')
    save_output_word_dicts_as_json(index_output, output_index, type_map, 'output_dicts.json')
    print(index_output)



