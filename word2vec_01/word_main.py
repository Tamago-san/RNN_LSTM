from gensim.models import KeyedVectors

embeddings_model = KeyedVectors.load_word2vec_format('/path/to/entity_vector.model.bin', binary=True)
