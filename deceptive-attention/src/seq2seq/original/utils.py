import pickle
from collections import defaultdict

import nltk

PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


class Language:
    def __init__(self, name):
        self.name = name
        self.w2i = defaultdict(lambda: len(self.w2i))

        # useful constants
        self.w2i['<pad>'] = PAD_token
        self.w2i['<sos>'] = SOS_token
        self.w2i['<eos>'] = EOS_token
        self.w2i['<unk>'] = UNK_token

        # index to word
        self.i2w = {v: k for k, v in self.w2i.items()}

    def get_index(self, word):
        idx = self.w2i[word]
        self.i2w[idx] = word
        return idx

    def stop_accepting_new_words(self):
        # set the w2i to return the unk token, upon seeing an unknown word
        self.w2i = defaultdict(lambda: UNK_token, self.w2i)

    def get_num_words(self):
        return len(self.w2i)

    def get_word(self, index):
        if index in self.i2w:
            return self.i2w[index]
        return self.i2w[UNK_token]

    def get_sent_rep(self, sentence):
        sentence = "<sos> " + sentence + " <eos>"
        # spaces are deliberate above

        return [self.get_index(w) for w in sentence.split()]

    def save_vocab(self, filename):
        # convert default dict to a normal dict as pickle 
        # would not be able to dump a default dict
        pickle.dump(dict(self.w2i), open(filename, 'wb'))

    def get_vocab_size(self):
        return len(self.i2w)

    def load_vocab(self, filename):
        vocab = pickle.load(open(filename, 'rb'))

        # copy the normal dict into default dict
        self.w2i = defaultdict(lambda: len(self.w2i), vocab)
        self.i2w = {v: k for k, v in self.w2i.items()}

    def pad_sequences(self, seqs, max_len):
        return [self.pad_sequence(seq, max_len) for seq in seqs]

    @staticmethod
    def pad_sequence(seq, max_len):
        padded_seq = seq + [PAD_token] * (max_len - len(seq))
        return padded_seq[:max_len]


def bleu_score_corpus(references, candidates, trg_lang, logger):
    """
    Computes the (test) corpus wide BLEU score using the library NLTK. Takes a list of reference sentences
    (target sentences in target language) as well as candidates, transforms then into the expected format for NLTK
    and computes the score.
    """

    # targets are indices --> converting indices to word tokens

    target_sentences = []
    for ref in references:
        # getting the target indices and cutting off <eof> and <sos> from target sentences
        trg_tokens = [trg_lang.get_word(w) for w in ref[2][0][1:-1]]
        target_sentences.append([trg_tokens])

    return bleu_score(target_sentences, candidates, logger)


def bleu_score(target_sentences, candidates, logger):
    """
    Computes the (test) corpus wide BLEU score using the library NLTK. Takes a list of reference sentences
    (target sentences in target language) as well as candidates, transforms then into the expected format for NLTK
    and computes the score.
    """

    assert len(candidates) == len(target_sentences)

    candidate_sentences = [candidate.split() for candidate in candidates]

    logger.info('\n')

    for i in range(len(candidate_sentences[:3])):
        logger.info(f'candidate {candidate_sentences[i]}')
        logger.info(f'reference {target_sentences[i]}')
        logger.info('\n')

    # Expected structure by NLTK
    # target Sentences: [[...], [...]]
    # candidate Sentences: [..., ...]

    return nltk.translate.bleu_score.corpus_bleu(target_sentences, candidate_sentences)
