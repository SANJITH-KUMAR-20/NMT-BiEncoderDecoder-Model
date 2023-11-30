import unicodedata
import re
import math

def UnicodetoASCii(input_sentence): # for Unicode to ASCII
    return "".join(i for i in unicodedata.normalize("NFD",input_sentence) if unicodedata.category(i)!="Mn")

def normalizeInput(input_sentence) -> str: # Normalizing text :removing punctuations,converting to lower case.removing excess spaces
    sent = re.sub(r" ##AT##-##AT## ", r" ",input_sentence)
    sent = UnicodetoASCii(sent.lower().strip())
    sent = re.sub(r"([.!?])",r" \1",sent)
    sent = re.sub(r"[^a-zA-Z.!?]+",r" ",sent)
    return sent

def filterhelper(sentences,max_length: int) -> bool: # returns True or False based on whether the given sentences are within the acceptable length
    filtered = len(sentences[0].split(" ")) < max_length and len(sentences[1].split(" "))<max_length
    return filtered

def filtersentences(sentences,max_len: int): # returns a pair of acceptable senteces
    return [sent for sent in sentences if filterhelper(sent,max_len)]
