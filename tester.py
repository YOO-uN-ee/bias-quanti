import codecs, os, json, operator, pickle
from random import shuffle
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import spatial
import math
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

def load_glove(path):
    with open(path) as f:
        lines = f.readlines()
    
    wv = []
    vocab = []
    for line in lines:
        tokens = line.strip().split(" ")
        try:
            assert len(tokens) == 301
            vocab.append(tokens[0])
            wv.append([float(elem) for elem in tokens[1:]])
        except:
            print(line)
        
    w2i = {w: i for i, w in enumerate(vocab)}
    wv = np.array(wv).astype(float)
    print(len(vocab), wv.shape, len(w2i))
    
    return wv, w2i, vocab

from sklearn.decomposition import PCA
def my_pca(wv):
    wv_mean = np.mean(np.array(wv), axis=0)
    wv_hat = np.zeros(wv.shape).astype(float)

    for i in range(len(wv)):
        wv_hat[i, :] = wv[i, :] - wv_mean

    main_pca = PCA()
    main_pca.fit(wv_hat)
#     main_pca.fit(wv)

    return main_pca

def simi(a, b):
    ret = 1-spatial.distance.cosine(a, b)
    return ret

wv_dd, w2i_dd, vocab_dd = load_glove('vectors_ddglove_gender.txt')
wv_glove, w2i_glove, vocab_glove = load_glove('vectors_glove.txt')

from utils import doPCA

female_ids = [43, 195, 509, 829, 1086, 1361, 1419, 1597, 2921, 3158, 6064, 7496, 7503, 13598, 13641, 13786, 14204, 18384, 19500, 19747, 20332, 22464, 22931, 24041, 25054, 26028, 26511, 26944, 29228, 29667]
male_ids = [19, 304, 653, 1251, 1557, 1634, 1642, 3795, 5645, 10071, 10212, 11895, 12151, 12730, 13807, 14653, 15311, 16259, 19026, 19255, 19478, 19728, 24812, 25269, 25550, 25553, 26031, 26791, 27600, 28077]
all_ids = female_ids + male_ids

definitional_pairs = [['she', 'he'], ['herself', 'himself'], ['her', 'his'], ['daughter', 'son'],
                      ['girl', 'boy'], ['mother', 'father'], ['woman', 'man'], ['mary', 'john'],
                      ['gal', 'guy'], ['female', 'male']]


old_gender_direction = doPCA(definitional_pairs, wv_glove, w2i_glove).components_[0]

my_gender_direction = sum([wv_glove[w_id, :] for w_id in female_ids]) / len(female_ids) - sum([wv_glove[w_id, :] for w_id in male_ids]) / len(male_ids)

male_specific = []
female_specific = []
with open('male_word_file.txt') as f:
    for l in f:
        if not l.strip() in vocab_dd:
            continue
        male_specific.append(l.strip())
with open('female_word_file.txt') as f:
    for l in f:
        if not l.strip() in vocab_dd:
            continue
        female_specific.append(l.strip())