import string
import pandas as pd
import numpy as np
import pickle

from com_1_edit_distance import com_1_edit_distance

def compute_channel_probability(x, y, err):
    all_letters_unicode_b10 = np.array([2947,2949,2950,2951,2952,2953,2954,2958,2959,2960,2962,2963,2964,2965,2969,2970,2972,2974,2975,2979,2980,2984,2985,2986,2990,2991,2992,2993,2994,2995,2996,2997,2998,2999,3000,3001,3006,3007,3008,3009,3010,3014,3015,3016,3018,3019,3020,3021])
    with open("./src/_count.pkl", "rb") as file:
        _count = pickle.load(file)

    count_array = np.load("./src/count_array.npy")
    sub = np.load("./src/sub.npy")
    tra = np.load("./src/tra.npy")
    ins = np.load("./src/ins.npy")
    _del = np.load("./src/del.npy")

    if err[1] == "insertion":
        i = np.where(all_letters_unicode_b10 == ord(y[err[2]-1]))[0][0]
        j = np.where(all_letters_unicode_b10 == ord(y[err[2]]))[0][0]
        
        return ins[i, j] / count_array[i]

    elif err[1] == "transposition":
        i = np.where(all_letters_unicode_b10 == ord(x[err[2]]))[0][0]
        j = np.where(all_letters_unicode_b10 == ord(x[err[3]]))[0][0]
        
        return tra[i, j] / _count[(i, j)]
    
    elif err[1] == "substitution":
        pre = np.where(all_letters_unicode_b10 == ord(x[err[2]]))[0][0]
        post = np.where(all_letters_unicode_b10 == ord(y[err[2]]))[0][0]

        return sub[pre, post] / count_array[post]
        
    
    elif err[1] == "deletion":
        i = np.where(all_letters_unicode_b10 == ord(x[err[2]-1]))[0][0]
        j = np.where(all_letters_unicode_b10 == ord(x[err[2]]))[0][0]
        
        return _del[i, j] / _count[(i, j)]
    
    else:
        raise Exception("Unkown Error Cause")


def channel_model_prob(sentence, tokens, non_words):

    wiki = pd.read_csv("./src/cleaned_wiktionary_words.csv", header=None)

    channel_probs = {}
    for non_word in non_words:
        for i in range(len(wiki)):
            res = com_1_edit_distance(wiki.iloc[i, 0], non_word)        
            if res is not None:
                key = wiki.iloc[i, 0] + "^" + non_word + "^" + str(res)
                channel_probs[key] = 0
    
    for key in channel_probs:
        x, y, err = key.split("^")
        err = eval(err)
        channel_probs[key] = compute_channel_probability(x, y, err)
    
    values_sum = 0
    for value in channel_probs.values():
        values_sum += value
    
    for key in channel_probs:
        channel_probs[key] = channel_probs[key] / values_sum
    
    return channel_probs
