from tqdm import tqdm

def read_vocabulary(fileName,stoken,etoken,utoken):
    vocab = dict()
    num_words = 0
    with open(fileName) as f:
        for line in f:
            word = line.strip().split()[0]
            if word not in vocab:
                vocab[word] = num_words
                num_words += 1
    for word in [stoken,etoken,utoken]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab,num_words

def read_sentences(fileName,vocab,stoken,etoken,utoken):
    orig_sent = list()
    sent = list()
    lines = [line for line in open(fileName)]
    for line in tqdm(lines,desc='Reading input sentences'):
        words = line.strip().split()
        if len(words)>0:
            orig_sent.append(" ".join(words))
            if words[0] != stoken:
                words.insert(0,stoken)
            if words[-1] != etoken:
                words.append(etoken)
            for jdx,word in enumerate(words):
                words[jdx] = (words[jdx] if word in vocab else utoken)
            sent.append(" ".join(words))

    return orig_sent, sent

def count_sequences(fileName,stoken,etoken,max_len):
    max_words = 0
    num_seq = 0
    lines = [line for line in open(fileName)]
    for line in tqdm(lines,desc='Reading input sentences'):
        words = line.strip().split()
        if len(words)>0:
            nwords = len(words)+int(words[0]!=stoken)+int(words[-1]!=etoken)
            if nwords <= max_len:
                if nwords > max_words:
                    max_words = nwords
                num_seq += 1
    return num_seq, max_words

