from collections import defaultdict

from tqdm import tqdm


def compute_frequencies(lines, top_k):
    """
    Compute frequencies of each word in 'lines', sort it and return a set of 'top_k' words.
    """
    # word 2 frequency
    w2f = defaultdict(lambda: 0.0)

    for idx, line in tqdm(enumerate(lines)):
        for word in line.split():
            w2f[word] += 1.0

    top_k_word_frequencies = sorted(w2f.items(), key=lambda x: -x[1])[:top_k]
    top_k_words = set([w[0] for w in top_k_word_frequencies])

    return top_k_words


def unkify_lines(lines, top_words):
    """
    For every word in 'lines' check if it is contained in the 'top_words', if so, use it if not replace it with
    <unk> token.
    """

    new_lines = []
    for line in tqdm(lines):
        new_words = []
        for word in line.split():
            if word in top_words:
                new_words.append(word)
            else:
                new_words.append("<unk>")

        new_lines.append(" ".join(new_words))

    return new_lines
