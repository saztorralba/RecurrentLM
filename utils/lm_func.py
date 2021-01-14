from tqdm import tqdm

#Function to read a vocabulary file and add special tokens
def read_vocabulary(lines=None,**kwargs):
    vocab = dict()
    num_words = 0
    if lines is None:
        if kwargs['vocabulary'] == None:
            return read_characters(**kwargs)
        else:
            lines = [line for line in open(kwargs['vocabulary'])]
    for line in lines:
        word = line.strip().split()[0]
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    for word in [kwargs['start_token'],kwargs['end_token'],kwargs['unk_token']]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab, False

#Function to get a vocabulary from all characters in input text
def read_characters(lines=None,**kwargs):
    vocab = dict()
    num_words = 0
    if lines is None:
        lines = [line for line in open(kwargs['input_file'])]
    for line in lines:
        line = line.strip()
        for char in line:
            if char not in vocab:
                vocab[char] = num_words
                num_words+=1
    for word in [kwargs['start_token'],kwargs['end_token'],kwargs['unk_token']]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab, True

#Read sentences, with and without special tokens
def read_sentences(lines=None,**kwargs):
    orig_sent = list()
    sent = list()
    if lines is None:
        lines = [line for line in open(kwargs['input_file'])]
    for line in tqdm(lines,desc='Reading input sentences',disable=(kwargs['verbose']<2)):
        if kwargs['characters']:
            words = list(line.strip())
        else:
            words = line.strip().split()
        if len(words)>0:
            orig_sent.append(words)
            if words[0] != kwargs['start_token']:
                words.insert(0,kwargs['start_token'])
            if words[-1] != kwargs['end_token']:
                words.append(kwargs['end_token'])
            for jdx,word in enumerate(words):
                words[jdx] = (words[jdx] if word in kwargs['vocab'] else kwargs['unk_token'])
            sent.append(words)

    return orig_sent, sent

#Obtain number of valid sentences and maximum length of them
def count_sequences(lines=None,**kwargs):
    max_words = 0
    num_seq = 0
    if lines is None:
        lines = [line for line in open(kwargs['input_file'])]
    for line in tqdm(lines,desc='Reading input sentences',disable=(kwargs['verbose']<2)):
        if kwargs['characters']:
            words = list(line.strip())
        else:
            words = line.strip().split()
        if len(words)>0:
            nwords = len(words)+int(words[0]!=kwargs['start_token'])+int(words[-1]!=kwargs['end_token'])
            if nwords <= kwargs['max_length']:
                if not 'min_length' in kwargs or nwords>=kwargs['min_length']:
                    if nwords > max_words:
                        max_words = nwords
                    num_seq += 1
    return num_seq, max_words

