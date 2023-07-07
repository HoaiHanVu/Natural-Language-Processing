import numpy as np
import pandas as pd
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from scipy import linalg
from collections import defaultdict


def tokenize(corpus):
    data = re.sub(r'[,!?;-]+', '.', corpus)
    data = nltk.word_tokenize(data)
    data = [char.lower() for char in data
           if char.isalpha()
           or char == '.'
           or emoji.get_emoji_regexp().search(char)
           ]
    return data

def get_windows(words, C):
    i = C
    while i < len(words) - C:
        center_word = words[i]
        context_words = words[i-C:i] + words[i+1:i+C+1]
        yield context_words, center_word
        i += 1

def get_dict(data):
    """
    Input:
        data: the data want to pull from
    Output:
        word2idx: returns dictionary mapping the word to its index
        idx2word: returns dictionary mapping the index to its word
    """
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    word2idx = {}
    idx2word = {}
    for k in words:
        word2idx[k] = idx
        idx2word[idx] = k
        idx += 1
    return word2idx, idx2word


def word_to_one_hot_vector(word, word2idx, V):
    """
    Input:
        word: the letter of corpus want to transform into one hot vector
        word2idx: dictionary with key is word and value is index of word in string
        V: size of vocabulary
    Output:
        one_hot_vector: a vector one hot of word
    """
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2idx[word]] = 1
    
    return one_hot_vector

def context_words_to_vector(context_words, word2idx, V):
    """
    Input:
        context_words: list of context words
        word2idx: dictionary with key is word and value is index of word in string
        V: size of vocabulary
    Output:
        context_words_vector: vectors of all context words
    """
    context_words_vectors = [word_to_one_hot_vector(w, word2idx, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    
    return context_words_vectors

def get_training_example(words, C, word2idx, V):
    for context_words, center_word in get_windows(words, C):
        yield context_words_to_vector(context_words, word2idx, V),\
              word_to_one_hot_vector(center_word, word2idx, V)
        
def get_idx(words, word2idx):
    idx = []
    for word in words:
        idx = idx + [word2idx[word]]
    return idx

def pack_idx_with_frequency(context_words, word2idx):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2idx)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed

def get_vectors(data, word2idx, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2idx[center_word]] = 1
        context_words = data[(i - C) : i] + data[(i + 1) : (i + C + 1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2idx):
            x[idx] = freq / num_ctx_words
        yield x, y
        i += 1
        if i >= len(data) - C:
#             print("i is being set to", C)
            i = C

def get_batches(data, word2idx, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2idx, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch_x = []
            batch_y = []

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0)
    
    return yhat

def compute_cost(y, yhat, batch_size):
    loss = np.sum(np.multiply(np.log(yhat), y))
    
    cost = -1/batch_size * loss
    cost = np.squeeze(cost)
    
    return cost

def initialize_model(N, V, random_seed=42):
    """
    N: dimensions of word vector
    V: number of words in vocabulary 
    """
    np.random.seed(random_seed)

    W1 = np.random.rand(N, V)
    W2 = np.random.rand(V, N)
    b1 = np.random.rand(N, 1)
    b2 = np.random.rand(V, 1)
    
    return W1, W2, b1, b2

def forward_prop(x, W1, W2, b1, b2):
    h = np.dot(W1, x) + b1
    h = relu(h)
    z = np.dot(W2, h) + b2
    
    return z, h

def back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size):
    grad_W1 = 1/batch_size * np.dot(relu(np.dot(W2.T, yhat - y)), x.T)
    grad_W2 = 1/batch_size * np.dot(yhat - y, h.T)
    grad_b1 = 1/batch_size * np.dot(relu(np.dot(W2.T, yhat -y)), np.ones((batch_size, 1)))
    grad_b2 = 1/batch_size * np.dot(yhat - y, np.ones((batch_size, 1)))
    
    
    
    
    return grad_W1, grad_W2, grad_b1, grad_b2

def gradient_descent(data, word2idx, N, V, num_iters, alpha, batch_sizes,
                     initialize_model=initialize_model, get_batches=get_batches, forward_prop=forward_prop,
                     softmax=softmax, compute_cost=compute_cost, back_prop=back_prop, random_seed=42):
    '''
    
      Inputs: 
        data:      text/list of words
        word2idx:  words to Indices
        N:         dimension of hidden vector  
        V:         dimension of vocabulary 
        num_iters: number of iterations  
        random_seed: random seed to initialize the model's matrices and vectors
        initialize_model: your implementation of the function to initialize the model
        get_batches: function to get the data in batches
        forward_prop: your implementation of the function to perform forward propagation
        softmax: your implementation of the softmax function
        compute_cost: cost function (Cross entropy)
        back_prop: your implementation of the function to perform backward propagation
     Outputs: 
        W1, W2, b1, b2:  updated matrices and biases after num_iters iterations

    '''
    W1, W2, b1, b2 = initialize_model(N, V, random_seed=random_seed)
    batch_size = batch_sizes
    
    iters = 0
    C = 2
    
    for x, y in get_batches(data, word2idx, V, C, batch_size):
        z, h = forward_prop(x, W1, W2, b1, b2)
        
        yhat = softmax(z)
        
        cost = compute_cost(y, yhat, batch_size)
        if (iters) % 10 == 0 and iters != 0:
            print('Iters: {} cost: {:.5f}'.format(iters, cost))
        
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size)
        
        W1 = W1 - alpha * grad_W1
        W2 = W2 - alpha * grad_W2
        b1 = b1 - alpha * grad_b1
        b2 = b2 - alpha * grad_b2
        
        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66
            
    return W1, W2, b1, b2