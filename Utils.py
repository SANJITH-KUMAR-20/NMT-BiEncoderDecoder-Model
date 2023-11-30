import LangObj as lg
import DataPrep_utils as Dp
import os
from random import shuffle
import math
import torch

def prepareLangObjs(LangSource,LangTarget,file_path1,file_path2):#Prepares source and target language classes
     lang1 = open(file_path1,encoding="utf-8").read().strip().split("\n")
     lang2 = open(file_path2,encoding="utf-8").read().strip().split("\n")

     if len(lang1)!=len(lang2):
          raise ValueError("Source and Target text sizes dont align")
     
     pairs = []
     for i,j in zip(lang1,lang2):
          pairs.append([Dp.normalizeInput(i),Dp.normalizeInput(j)])
     LangSource = lg.Language(LangSource)
     LangTarget = lg.Language(LangTarget)
     return LangSource,LangTarget,pairs

def prepareData(LangSource,LangTarget,Filepath1,Filepath2,max_vocab_size = 100000,
                prec_train_set = 0.9,trim = 0,shuff = True): #Preparing and Splitting Data
     
     Source,target,pairs = prepareLangObjs(LangSource,LangTarget,Filepath1,Filepath2)
     
     if trim:
          pairs = Dp.filtersentences(pairs,trim)
     
     for i in pairs:
          Source.countSent(i[0])
          target.countSent(i[1])
          #error:pairs = [(Source.AddSentences(i[0]),target.AddSentences(i[1]))]
     Source.CheckMaxVocabSize(max_vocab_size)
     target.CheckMaxVocabSize(max_vocab_size)
     pairs = [(Source.AddSentences(pair[0]),target.AddSentences(pair[1])) for pair in pairs]

     if shuff:
        shuffle(pairs)

     train_pairs = pairs[:math.ceil(len(pairs)*prec_train_set)]
     test_pairs = pairs[math.ceil(len(pairs)*prec_train_set):]
     print("Train: {}".format(len(train_pairs)))
     print("Test: {}".format(len(test_pairs)))
     return Source,target,train_pairs,test_pairs

def indexesFromSentence(lang,sentence):# Converts sentenses to their indexes
    indexes = []
    for word in sentence.split(" "):
         try:
              indexes.append(lang.word_to_idx[word])
         except:
              indexes.append(lang.word_to_idx["UNK"])
    return indexes

def tensorFromSentence(language,sentence):# converts indexed sentenses to tensors
     indexes = indexesFromSentence(language,sentence)
     indexes.append(lg.EOS)
     result = torch.LongTensor(indexes).view(-1)
     if torch.cuda.is_available():
          return result.cuda()
     else:
          return result
     
def tensorFromPair(SourceLang,TargetLang,sentences): #pairs the inp and out sentences
     input = tensorFromSentence(SourceLang,sentence=sentences[0])
     output = tensorFromSentence(TargetLang,sentences[1])
     return (input,output)

def sentenceFromTensor(lang,tensor): # converts tensors to sentences
     if tensor:
        raw = tensor.raw
        sent = ""
        for i in raw[:len(raw)]:
            sent+= lang.idx_to_word[i.item()] + " "
        sent+=lang.idx_to_word[raw[-1].item()]
        return sent
     else:
        return ""

def batchify(data, input_lang, output_lang, batch_size, shuffle_data=True):#Batches data based on batch size
    if shuffle_data == True:
        shuffle(data)
    number_of_batches = len(data) // batch_size
    batches = list(range(number_of_batches))
    longest_elements = list(range(number_of_batches))
    
    for batch_number in range(number_of_batches):
        longest_input = 0
        longest_target = 0
        input_variables = list(range(batch_size))
        target_variables = list(range(batch_size))
        index = 0      
        for pair in range((batch_number*batch_size),((batch_number+1)*batch_size)):
            input_variables[index], target_variables[index] = tensorFromPair(input_lang, output_lang, data[pair])
            if len(input_variables[index]) >= longest_input:
                longest_input = len(input_variables[index])
            if len(target_variables[index]) >= longest_target:
                longest_target = len(target_variables[index])
            index += 1
        batches[batch_number] = (input_variables, target_variables)
        longest_elements[batch_number] = (longest_input, longest_target)
    return batches , longest_elements, number_of_batches



def pad_batch(batch):#pads batches to allow for sentences of variable lengths to be computed in parallel
    padded_inputs = torch.nn.utils.rnn.pad_sequence(batch[0],padding_value=lg.EOS)
    padded_targets = torch.nn.utils.rnn.pad_sequence(batch[1],padding_value=lg.EOS)
    return (padded_inputs, padded_targets)