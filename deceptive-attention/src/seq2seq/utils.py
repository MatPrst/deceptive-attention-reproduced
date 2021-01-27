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


def get_target_sentences_as_list(data_batch, trg_lang):
    target_sentences = []
    for data in data_batch:
        # getting the target indices and cutting off <eof> and <sos> from target sentences
        trg_tokens = [trg_lang.get_word(w) for w in data[2][0][1:-1]]
        target_sentences.append([trg_tokens])
    return target_sentences


def bleu_score_nltk(target_sentences, candidates):
    """
    Computes the (test) corpus wide BLEU score using the library NLTK. Takes a list of reference sentences
    (target sentences in target language) as well as candidates, transforms then into the expected format for NLTK
    and computes the score.
    """

    assert len(candidates) == len(target_sentences)

    # candidates are already words
    # targets are indices --> converting indices to word tokens

    # target_sentences = get_target_sentences_as_list(references, trg_lang)
    # for ref in references:
    #     # getting the target indices and cutting off <eof> and <sos> from target sentences
    #     trg_tokens = [trg_lang.get_word(w) for w in ref[2][0][1:-1]]
    #     target_sentences.append([trg_tokens])
    candidate_sentences = [candidate.split() for candidate in candidates]

    for i in range(len(candidate_sentences[:10])):
        print('candidate ', candidate_sentences[i])
        print('reference ', target_sentences[i])
        print('\n')

    # store files for computing BLEU score for compare-mt afterwards manually via the terminal
    # references_file_path = f"{out_path}.ref.eng"
    # candidates_file_path = f"{out_path}.cand.de"
    #
    # with open(references_file_path + '', 'w') as file_handler:
    #     file_handler.write("\n".join(' '.join(item) for item in target_sentences))
    #
    # with open(candidates_file_path + '', 'w') as file_handler:
    #     file_handler.write("\n".join(str(item) for item in candidates))

    # Expected structure by NLTK
    # target Sentences: [[...], [...]]
    # candidate Sentences: [..., ...]

    return nltk.translate.bleu_score.corpus_bleu(target_sentences, candidate_sentences)