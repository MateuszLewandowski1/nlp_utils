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


def text_cleaner(string2):
    from cleantext import clean
    return clean(string2, no_urls=True, no_digits=True, no_line_breaks=True, no_emails=True, no_phone_numbers=True,
                 no_numbers=True, no_currency_symbols=True, no_punct=True, replace_with_digit="", replace_with_url="",
                 replace_with_email="", replace_with_currency_symbol="", replace_with_number="",
                 replace_with_phone_number="")



if __name__ == '__main__':
    pass

from typing import Callable, Optional, Tuple, Union