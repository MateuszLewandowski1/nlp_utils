import pandas as pd
from gensim.models import word2vec, Word2Vec
from nltk.tokenize import wordpunct_tokenize
import transformers
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Input, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from gensim.models import FastText
from tfrecordds import create_dataset
import tensorflow as tf
from sklearn.metrics import f1_score


def data_loader():
    path = '/mnt/sd1/DS/jigsaw/jigsaw-toxic-comment-train-processed-seqlen128.csv'
    table = pd.read_csv(path)
    return wordpunct_tokenize("".join(table.comment_text.to_list()))


def clean_string(string1):
    """removes any trash from the string"""
    import string
    digits = '1234567890'
    string1 = text_cleaner(string1)
    cleaned = ''.join(ch for ch in string1 if ch not in (string.punctuation + digits)).splitlines()  # possible a list,
    # need to collapse it
    str1 = ""
    for elem in cleaned:
        str1 += elem
    return str1.replace(" ", "")


def get_word_embedding(word):  # base idea to get the embeddings
    model = Word2Vec.load("word2vec_1.model")
    return model.wv[list(word)]


def get_fast_text_embedding(word, model):  # only if there is a word out of vocab fow word 2 vec
    return model.wv[list(word)]

# get_word_embedding(list(clean_string(table.comment_text[15]))).shape


def text_cleaner(string2):
    from cleantext import clean
    return clean(string2, no_urls=True, no_digits=True, no_line_breaks=True, no_emails=True, no_phone_numbers=True,
                 no_numbers=True, no_currency_symbols=True, no_punct=True, replace_with_digit="", replace_with_url="",
                 replace_with_email="", replace_with_currency_symbol="", replace_with_number="",
                 replace_with_phone_number="")


def train_word_2_vec_words(text):

    model = FastText([text], size=100, window=5, min_count=1, workers=4)
    model.save("/mnt/sd1/DS/ex_model.model")

    # command training the word2vec on words
    # train_word_2_vec_words((text_cleaner(table.comment_text.to_list())).split(" "))





def make_tf_record(input1):
    """takes input 1 and stores it into a td dataset / tf record object"""
    pass


def store_to_npy(table):
    path_to_store = '/mnt/sd1/DS/word2vec_embedding.npy'
    lista = []
    for i in range(10):
        lista.append(get_word_embedding((text_cleaner(table.comment_text[i]).split(" "))))
    np.save(path_to_store, lista)


def labels_to_npy(table, size=100):
    path_to_store = '/mnt/sd1/DS/labels.npy'
    lista = []
    for i in range(10):
        # print(i)
        lista.append(table.toxic[i])
    np.save(path_to_store, lista)



def prepare_features_using_bert():
    pass


def train_generator(table=pd.read_csv('/mnt/sd1/DS/jigsaw/jigsaw-toxic-comment-train-processed-seqlen128.csv',
                                      nrows=178832),
                    model=FastText.load("/mnt/sd1/DS/word2vec_fast_text.model")):
    count = 0
    while True:
        x_train = get_fast_text_embedding((text_cleaner(table.comment_text.to_list()[count])).split(" "), model)
        y_train = np.sign(table.toxic[count] + table.severe_toxic[count] + table.obscene[count] + table.threat[count] +
                          table.insult[count])
        count += 1
        if y_train == 1:
            y_train = 1
        else:
            y_train = 0

        x_train = np.asarray(x_train).astype('float32').reshape((1, -1, 100))
        y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
        yield x_train, y_train


def validation_generator(table=pd.read_csv('/mnt/sd1/DS/jigsaw/jigsaw-toxic-comment-train-processed-seqlen128.csv',
                                           skiprows=178833, names=['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat',
                                                                    'insult', 'identity_hate', 'input_word_ids', 'input_mask',
                                                                    'all_segment_id']),
                    model=FastText.load("/mnt/sd1/DS/word2vec_fast_text.model")):
    count = 0
    while True:
        x_train = get_fast_text_embedding((text_cleaner(table.comment_text.to_list()[count])).split(" "), model)
        y_train = np.sign(table.toxic[count] + table.severe_toxic[count] + table.obscene[count] + table.threat[count] +
                          table.insult[count])
        count += 1
        if y_train == 1:
            y_train = 1
        else:
            y_train = 0

        x_train = np.asarray(x_train).astype('float32').reshape((1, -1, 100))
        y_train = np.asarray(y_train).astype('float32').reshape((-1, 1))
        yield x_train, y_train


def define_model():
    input1 = Input(shape=(None, 100), batch_size=1)
    rnn = LSTM(32, return_sequences=True, name='lstm1')(input1)
    rnn = LSTM(16, return_sequences=True, name='lstm2')(rnn)
    # rnn = tf.reshape(rnn, [-1, 1])
    # rnn = Dense(32, activation='relu')(input1)
    # rnn = Dense(64, activation='relu')(rnn)
    out = Dense(1, activation='sigmoid')(rnn)
    model = Model(inputs=input1, outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.summary()
    return model


if __name__ == '__main__':
    # path = '/mnt/sd1/DS/jigsaw/jigsaw-toxic-comment-train-processed-seqlen128.csv'
    # table = pd.read_csv(path)
    model_keras = define_model()
    # model = FastText.load("/mnt/sd1/DS/word2vec_fast_text.model")

    # ds_counter = tf.data.Dataset.from_generator(train_generator, output_types=tf.float32)
    # for count_batch in ds_counter.repeat().batch(1).take(1):
    #     print(count_batch.shape)
    #
    model_keras.fit_generator(train_generator(), steps_per_epoch=0.8*223540/5, epochs=5, verbose=1,
                              validation_data=validation_generator(), validation_steps=8941)

    model_keras.save('/mnt/sd1/DS/keras_models/model_keras_jigsaw_fast_text.h5')

