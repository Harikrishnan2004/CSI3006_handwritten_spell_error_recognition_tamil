import pandas as pd
import unicodedata

class_to_unicode = pd.read_csv("D:/hari files/project/sf_project/sf_project/src/class_to_unicode.csv")

def get_tamil_character(class_number):
    unicode_text = class_to_unicode[class_to_unicode["class"] == class_number]['unicode'].values
    
    if len(unicode_text) == 0:
        print("Class number not found.")
        return
    
    unicode_text_decoded = [bytes(text, 'utf-8').decode('unicode_escape') for text in unicode_text]
    
    tamil_characters = []
    for text in unicode_text_decoded:
        tamil_text = unicodedata.normalize("NFKD", text).encode("utf-8", "ignore").decode("utf-8")
        tamil_characters.append(tamil_text)
    
    print(tamil_characters[0])
    return tamil_characters[0]

