from gensim.models import FastText
import tensorflow as tf
import pandas
import transformers
import os
import numpy as np
from tokenizers import BertWordPieceTokenizer
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tqdm


class Embedd:
    """different word embeddings for ease of trying different approaches methods. included bert, some semantic
    embeddings (word2vec, fast text, tf-idf and keras embedding layer. path to a csv file with text in one of the columns"""

    def __init__(self, path_train, path_valid):
        self.path_train = path_train
        self.path_valid = path_valid
        self.table_train = pd.read_csv(self.path_train)
        self.table_valid = pd.read_csv(self.path_valid)


    def bert(self):
        train1 = pd.read_csv(self.path_train)
        valid = pd.read_csv(self.path_valid)

        def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
            """
            Encoder for encoding the text into sequence of integers for BERT Input
            """
            tokenizer.enable_truncation(max_length=maxlen)
            tokenizer.enable_padding(max_length=maxlen)
            all_ids = []

            for i in tqdm(range(0, len(texts), chunk_size)):
                text_chunk = texts[i:i + chunk_size].tolist()
                encs = tokenizer.encode_batch(text_chunk)
                all_ids.extend([enc.ids for enc in encs])

            return np.array(all_ids)

        AUTO = tf.data.experimental.AUTOTUNE

        # Configuration
        EPOCHS = 3
        BATCH_SIZE = 16
        MAX_LEN = 192

        # First load the real tokenizer
        tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
        # Save the loaded tokenizer locally
        tokenizer.save_pretrained('.')
        # Reload it with the huggingface tokenizers library
        fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)

        x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
        x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)

        y_train = train1.toxic.values
        y_valid = valid.toxic.values

        train_dataset = (
            tf.data.Dataset
                .from_tensor_slices((x_train, y_train))
                .repeat()
                .shuffle(2048)
                .batch(BATCH_SIZE)
                .prefetch(AUTO)
        )

        valid_dataset = (
            tf.data.Dataset
                .from_tensor_slices((x_valid, y_valid))
                .batch(BATCH_SIZE)
                .cache()
                .prefetch(AUTO)
        )


        def build_model(transformer, max_len=512):
            """
            function for training the BERT model
            """
            input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
            sequence_output = transformer(input_word_ids)[0]
            cls_token = sequence_output[:, 0, :]
            out = Dense(1, activation='sigmoid')(cls_token)

            model = Model(inputs=input_word_ids, outputs=out)
            model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

            return model

        transformer_layer = (
            transformers.TFDistilBertModel
                .from_pretrained('distilbert-base-multilingual-cased')
        )
        model = build_model(transformer_layer, max_len=MAX_LEN)

        model.summary()

        n_steps = x_train.shape[0] // BATCH_SIZE
        train_history = model.fit(
            train_dataset,
            steps_per_epoch=n_steps,
            validation_data=valid_dataset,
            epochs=EPOCHS
        )

        pass

    def tf_idf(self, word):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([self.table_train, self.table_train])
        feature_names = vectorizer.get_feature_names()
        dense = vectors.todense()
        denselist = dense.tolist()
        df = pd.DataFrame(denselist, columns=feature_names)
        return df

    def train_fast_text(self, text, path_to_save):
        """"""
        model = FastText([text], size=100, window=5, min_count=1, workers=4)
        model.save(f'{path_to_save}_fasttext.model')

    def fast_text(self, word, model):
        """

        :param word:
        :param model: model saved by the method "train_fast_text"
        :return:
        """
        return model.wv[list(word)]

    def keras_embedding_layer(self):
        pass


import pandas as pd
def text_cleaner(string2, ret_as_string=False):
    from cleantext import clean
    from nltk.corpus import stopwords
    cleaned_string = clean(string2, no_urls=True, no_digits=True, no_line_breaks=True, no_emails=True, no_phone_numbers=True,
                 no_numbers=True, no_currency_symbols=True, no_punct=True, replace_with_digit="", replace_with_url="",
                 replace_with_email="", replace_with_currency_symbol="", replace_with_number="",
                 replace_with_phone_number="")

    filtered_sentence = [w for w in cleaned_string.split(" ") if not w in stopwords.words('english')]
    # optional - do not return it as a list of separate words, but as a string
    if ret_as_string:
        str_to_rtrn = ''
        for elem in filtered_sentence:
            str_to_rtrn = elem + ' '
        return str_to_rtrn
    else:
        return filtered_sentence

import pandas as pd
path = '/mnt/sd1/DS/jigsaw/jigsaw-toxic-comment-train-processed-seqlen128.csv'
table = pd.read_csv(path, nrows=1000,usecols=['comment_text', 'toxic'])


# str1 = []
# for elem in table.comment_text.to_list():
#     str1.extend(text_cleaner(elem))
#
#
# ilosc_elem = len(list(set(str1)))  # get the number of unique words


# dict1 = dict(zip(list(set(str1)), range(0, 9326)))





def data_loader(path='/mnt/sd1/DS/tvn_zdrowie/tvn_zdrowie_train_data_classification_links_categories_3k.csv'):
    table = pd.read_csv(path, usecols=['Meta Description 1', '2_level 1'])  # feature, label
    # number_of_classes = table['2_level 1'].nunique()
    # dict1 = dict(zip(table['2_level 1'].unique().tolist(), range(0, number_of_classes)))  # converts labels to numbers
    # not needed for fast text
    # table['labels'] = table['2_level 1'].map(dict1, na_action='ignore')
    # table = table.drop(columns=['2_level 1'])
    table.columns = ['feature', 'label']
    return table



from numpy import array
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
cleaned = []
for i in range(table.shape[0]):
    cleaned.append(" ".join(text_cleaner(table.comment_text.to_list()[i])))
table['cleaned'] = pd.Series(cleaned)

docs = table['cleaned'].to_list()
labels = table.toxic.to_numpy()
# integer encode the documents
vocab_size = 9326
encoded_docs = [one_hot(d, vocab_size) for d in docs]

max_length = 578
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# print(padded_docs)
# define the model
from tensorflow.keras import layers
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K


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


model = Sequential()
model.add(Embedding(vocab_size, 64, input_length=max_length))
model.add(layers.LSTM(64))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=1, validation_split=0.2)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

