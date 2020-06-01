import pandas as pd
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
import tensorflow as tf
import datetime

def text_cleaner(string2):

    polish_stopwords = pd.read_csv('/home/erazer/PycharmProjects/jigsaw/nlp_utils/polish_stopwords.txt', header=None,
                                   names=['words']).words.to_list()
    from cleantext import clean
    cleaned_string = clean(string2, no_urls=True, no_digits=True, no_line_breaks=True, no_emails=True, no_phone_numbers=True,
                 no_numbers=True, no_currency_symbols=True, no_punct=True, replace_with_digit="", replace_with_url="",
                 replace_with_email="", replace_with_currency_symbol="", replace_with_number="",
                 replace_with_phone_number="")

    filtered_sentence = [elem for elem in cleaned_string.split(' ') if elem not in polish_stopwords]
    # optional - do not return it as a list of separate words, but as a string
    return filtered_sentence


def data_loader(path='/mnt/sd1/DS/tvn_zdrowie/tvn_zdrowie_train_data_classification_links_categories_3k.csv'):
    table = pd.read_csv(path, usecols=['Meta Description 1', '2_level 1'])  # feature, label
    number_of_classes = table['2_level 1'].nunique()
    dict1 = dict(zip(table['2_level 1'].unique().tolist(), range(0, number_of_classes)))  # converts labels to numbers
    table['labels'] = table['2_level 1'].map(dict1, na_action='ignore')
    table = table.drop(columns=['2_level 1'])
    table.columns = ['feature', 'label']
    return table


table = data_loader()

from tensorflow.keras.utils import to_categorical  # for one hot encoding for keras


def number_of_unique_words_in_comments(table):
    """ gives information about the size of an embedding - each word needs to have a unique hash"""
    str1 = []
    for elem in table.feature.to_list():
        str1.extend(text_cleaner(elem))

    ilosc_elem = len(list(set(str1)))
    return ilosc_elem


def find_longest_sequence(table):
    """parameter for an Embedding layer"""
    maksimum = 0
    for i in range(table.shape[0]):
        if len(text_cleaner(table.feature[i])) > maksimum:
            maksimum = len(text_cleaner(table.feature[i]))
    return maksimum

maksimum = find_longest_sequence()

ilosc_elem = number_of_unique_words_in_comments(table)

cleaned = []
for i in range(table.shape[0]):
    cleaned.append(" ".join(text_cleaner(table.feature.to_list()[i])))
table['cleaned'] = pd.Series(cleaned)

docs = table['cleaned'].to_list()
labels = table.label.to_numpy()
# integer encode the documents
vocab_size = ilosc_elem
encoded_docs = [one_hot(d, vocab_size) for d in docs]

max_length = maksimum
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model = Sequential()
model.add(Embedding(vocab_size, max_length))
# model.add(layers.LSTM(32))
model.add(layers.LSTM(16))
model.add(Flatten())
model.add(Dense(60, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(Dense(20, activation='softmax'))
# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[f1])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, to_categorical(labels), epochs=1000, verbose=1, validation_split=0.3, batch_size=100,
          callbacks=[tensorboard_callback]
          )
# evaluate the model
# loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
# print('Accuracy: %f' % (accuracy*100))

