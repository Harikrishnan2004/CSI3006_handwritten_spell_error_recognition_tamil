from channel_model import channel_model_prob
from language_model import language_model_prob

import string
import pandas as pd

sentence = "மளை பெய்தால், விவசாயிகளுக்கு அதிக வருமானம் கடைக்கும்"
# sentence = "மலை பெய்தால், விவசாயிகளுக்கு அதிக வருமானம் கிடைக்கும்"
# sentence = "எப்டி இருக்கிறீர்கள்? நான் நன்றாக இரக்கிறேன்"
# sentence = "நான் வாக்ளித்தேன், இந்த தேர்தலில் நங்கள் வாக்களித்தீர்களா?"
# sentence = "நடகை தம்மானா இனறு காலை அமெரிக்கா சென்றடைந்தார்"
# sentence = "துன்பம் நேரும் சமயத்தில் அதை கன்டு சிரிக்கப் பழகுங்கள்"

print("Input:\t", sentence)
print("Preprocessing...")

tokens = sentence.split(" ")
_tokens = tokens.copy()
for i in range(len(tokens)):
    _token = list(tokens[i])
    __token = _token.copy()
    for unit in _token:
        if unit in string.punctuation:
            __token.remove(unit)
    
    _tokens[i] = "".join(__token)

words_with_root = pd.read_csv("./src/MostusedRootwords.csv", header=None)
proper_nouns = pd.read_csv("./src/proper_nouns.csv", header=None)

non_words = []
for token in _tokens:
    if not words_with_root[2].str.split().apply(lambda x: any(token in word for word in x)).any() and not proper_nouns[0].str.contains(token).any():
        non_words.append(token)

print("Tokens   :", _tokens)
print("Non Words:", non_words)

if non_words != []:
    print("\nComputing Channel Model Probabilities...")
    channel_model_probabilities = channel_model_prob(sentence, _tokens, non_words)
    channel_model_probabilities = {k: v for k, v in sorted(channel_model_probabilities.items(), key=lambda element: element[1], reverse=True)}
    for key in channel_model_probabilities:
        print("{:<60}\t{:.15f}".format(key, channel_model_probabilities[key]))

    print("\nComputing Language Model Probabilities...")
    language_model_probabilities = language_model_prob(sentence, _tokens, non_words, channel_model_probabilities)
    language_model_probabilities = {k: v for k, v in sorted(language_model_probabilities.items(), key=lambda element: element[1], reverse=True)}
    for key in language_model_probabilities:
        print("{:<60}\t{:.15f}".format(key, language_model_probabilities[key]))

    combined_model = {}
    for key in channel_model_probabilities:
        combined_model[key] = channel_model_probabilities[key] * language_model_probabilities[key]

    print("\nCombined Model results")
    combined_model = {k: v for k, v in sorted(combined_model.items(), key=lambda element: element[1], reverse=True)}
    for key in combined_model:
        print("{:<60}\t{:.15f}".format(key, combined_model[key]))

    print("\nThe Non-Word errors, can be suggested with the following")
    extracted = []
    for non_word in non_words:
        for key in combined_model:
            if non_word in key and non_word not in extracted:
                print(non_word, "\t", key.split("^")[0])
                _tokens[_tokens.index(non_word)] = key.split("^")[0]
                sentence
                extracted.append(non_word)

    print("\nCorrected [Non Word error] Sentence:", " ".join(_tokens))

import fasttext
import numpy as np

model_path = "/home/arjun/Desktop/Winter24/Web Mining/PJ/src/cc.ta.300.bin"
model = fasttext.load_model(model_path)

print("\nFinding words that are in dictionary, but could fail the context")
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm_vector1 * norm_vector2)
    
    return similarity

def get_sentence_vector(sentence):
    vec = np.zeros(300)
    for token in sentence:
        vec += model.get_word_vector(token)
    
    return vec / len(sentence)

context_match = {}
for token in _tokens:
    tokens_copy = _tokens.copy()
    tokens_copy.remove(token)
    sentence_vec = get_sentence_vector(tokens_copy)
    token_vec = model.get_word_vector(token)

    context_match[token] = cosine_similarity(token_vec, sentence_vec)

context_match = {k: v for k, v in sorted(context_match.items(), key=lambda element: element[1])}
for token in context_match:
    print("{:<20}\t{:.15f}".format(token, context_match[token]))