import numpy as np
import pandas as pd
import tensorflow as tf

# from csv

train_path_file = ''
test_path_file = ''


def get_dataset(file_path, LABEL_COLUMN, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,
        label_name=LABEL_COLUMN,
        na_value='?',
        num_epochs=1,
        ignore_errors=True,
        **kwargs
    )
    return dataset


raw_train_data = get_dataset(train_path_file)
raw_test_data = get_dataset(test_path_file)

# numpy and pandas

# from tendor slices - numpy













