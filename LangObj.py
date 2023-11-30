SOS = 0 #Marks start of a sentence
EOS = 1 #Marks end of a sentence
PAD = 2 #Marks Unknown words i.e words not in vocabulary

#class LangeTypes:


class Language:
    def __init__(self,Language_name: str):
        self.language = Language_name
        self.word_to_idx = {"SOS":SOS,"EOS":EOS,"UNK":PAD}
        self.idx_to_word = {SOS:"SOS",EOS:"EOS",PAD:"UNK"}
        self.vocab_count = {}
        self.vocab_size = 3
        self.upperlimit = 0

    def countSent(self,sentence) -> None: #Helper function for the below countvocab function
        for i in sentence.split(" "):
            self.CountVocab(i)
    
    def CountVocab(self,word: str) -> None: #To populate the vocab_count dictionary which contians each word with it's count
        if word not in self.vocab_count:
            self.vocab_count[word] = 1
        else:
            self.vocab_count[word]+=1

    def CheckMaxVocabSize(self,max_vocab_size: int) -> None:#if the number of unique words in the dataset is larger than the specified max_vocab_size, creates an upper bar that is used to leave infrequent words out of the vocabulary 
        word_frequency = list(self.vocab_count.values())
        word_frequency.sort(reverse=True)
        if len(word_frequency)>max_vocab_size:
            self.upperlimit = word_frequency[max_vocab_size]

    def AddWord(self,word: str) -> str:#Adds words to the word_to_idx and idx_to_word dictionaries
        if self.vocab_count[word] > self.upperlimit:
            if word not in self.word_to_idx:
                self.word_to_idx[word] = self.vocab_size
                self.idx_to_word[self.vocab_size] = word
                self.vocab_size+=1
            return word
        else:
            return self.idx_to_word[2]
    
    
    def AddSentences(self,sentence: str) -> str: #adds unique index for unique words
        sent = ""
        for word in sentence.split(" "):
            unk_word = self.AddWord(word)
            if not sent:
                sent = unk_word
            else:
                sent = sent + " " +unk_word
        return sent