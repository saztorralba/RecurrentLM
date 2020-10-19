from tqdm import tqdm

#Function to read a vocabulary file and add special tokens
def read_vocabulary(args):
    if args.vocabulary == None:
        return read_characters(args)
    vocab = dict()
    num_words = 0
    with open(args.vocabulary) as f:
        for line in f:
            word = line.strip().split()[0]
            if word not in vocab:
                vocab[word] = num_words
                num_words += 1
    for word in [args.start_token,args.end_token,args.unk_token]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab, False

#Function to get a vocabulary from all characters in input text
def read_characters(args):
    vocab = dict()
    num_words = 0
    with open(args.input_file) as f:
        for line in f:
            line = line.strip()
            for char in line:
                if char not in vocab:
                    vocab[char] = num_words
                    num_words+=1
    for word in [args.start_token,args.end_token,args.unk_token]:
        if word not in vocab:
            vocab[word] = num_words
            num_words += 1
    return vocab, True

#Read sentences, with and without special tokens
def read_sentences(args):
    orig_sent = list()
    sent = list()
    lines = [line for line in open(args.input_file)]
    for line in tqdm(lines,desc='Reading input sentences'):
        if args.characters:
            words = list(line.strip())
        else:
            words = line.strip().split()
        if len(words)>0:
            orig_sent.append(words)
            if words[0] != args.start_token:
                words.insert(0,args.start_token)
            if words[-1] != args.end_token:
                words.append(args.end_token)
            for jdx,word in enumerate(words):
                words[jdx] = (words[jdx] if word in args.vocab else args.unk_token)
            sent.append(words)

    return orig_sent, sent

#Obtain number of valid sentences and maximum length of them
def count_sequences(args):
    max_words = 0
    num_seq = 0
    lines = [line for line in open(args.input_file)]
    for line in tqdm(lines,desc='Reading input sentences'):
        if args.characters:
            words = list(line.strip())
        else:
            words = line.strip().split()
        if len(words)>0:
            nwords = len(words)+int(words[0]!=args.start_token)+int(words[-1]!=args.end_token)
            if nwords <= args.max_length:
                if nwords > max_words:
                    max_words = nwords
                num_seq += 1
    return num_seq, max_words

