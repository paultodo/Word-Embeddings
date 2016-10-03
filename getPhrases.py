from gensim.models import phrases


# sentence stream : an iterable, with each value a list of token strings
bigram = Phrases(sentence_stream)
trigrma = Phrases(bigram[sentence_stream])