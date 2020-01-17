# -*- coding: utf-8 -*-
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

#%%

import sys
import os
import argparse
import zipfile
import collections

from tempfile import gettempdir
from six.moves import urllib
from six.moves import urllib

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
print(current_path)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir', 
    type=str, 
    default=os.path.join(current_path, 'log'),
    help='The log directory for TensorBoard summaries.'
)

FLAGS, unparsed = parser.parse_known_args()

print(FLAGS, '\n', FLAGS.log_dir, '\n', unparsed, '\n', parser)

if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)


# ---------------------------------------------Step 1: Download the data.-------------------------------------------

url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """
    Download a file if not present, and make sure it's right size.
    """
    # 文件路径 + 文件名称的拼接
    local_filename = os.path.join(gettempdir(), filename)

    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url+filename, local_filename)

    statinfo = os.stat(local_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify '+ local_filename + '. Can you get to it with a browser?')
    return local_filename

filename = maybe_download('text8.zip', 31344016)



# Read the data into a list of strings.

def read_data(filename):
    """
    Extract the first file enclosed in a zip file as a list of words.
    """
    with zipfile.ZipFile(filename) as f:
        # as_str()： 将字节或 unicode 转换为 bytes,使用 UTF-8 编码进行文本处理.
        # as_bytes()： 将字节或 unicode 转换为 bytes,使用 UTF-8 编码进行文本处理.
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

vocabulary = read_data(filename)
print('Data Size: ', len(vocabulary))



# ----------------------Step 2: Build the dictionary and replace rare words with UNK token.-------------------------
vocabulary_size = 50000

def build_datasets(words, n_words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0

    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data,count, dictionary, reversed_dictionary




# def multipliers():
#   return [lambda x : i * x for i in range(4)]

# for p in multipliers():
#     print(p(2))



