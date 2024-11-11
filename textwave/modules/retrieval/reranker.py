import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
import torch
import numpy as np

class Reranker:
    """
    A class to perform reranking of documents based on their relevance to a given query
    using three possible approaches: cross-encoder, TF-IDF, or a hybrid of both.
    """

    def __init__(self, type, corpus, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the Reranker class with specified reranking type and model name.

        :param type: A string indicating the type of reranking ('cross_encoder', 'tfidf', 'hybrid', 'sequential', or 'tfidf_corpus').
        :param cross_encoder_model_name: A string specifying the cross-encoder model name (default is 'cross-encoder/ms-marco-MiniLM-L-6-v2').
        """
        self.type = type
        self.corpus = corpus
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)

        # Initialize vetorization of corpus
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_corpus_matrix = self.vectorizer.fit_transform(self.corpus)

    def rerank(self, query, context, distance_metric="cosine"):
        """
        Selects the reranking method based on the initialized type.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document to be reranked.
        :param distance_metric: A string indicating the distance metric to use for TF-IDF reranking (default is "cosine").
        :return: Ranked documents, indices, and scores based on the selected reranking method.
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "sequential":
            return self.sequential_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "tfidf_corpus":
            return self.tfidf_corpus_rerank(query, distance_metric=distance_metric)

    def cross_encoder_rerank(self, query, context):
        """
        Reranks documents based on relevance to the query using a cross-encoder model.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :return: A list of ranked documents, their indices, and relevance scores.
        """
        if len(context) == 1:
            return context, [0], [1.0]
        
        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            relevance_scores = logits.squeeze().tolist()

        ranked_indices = torch.argsort(torch.tensor(relevance_scores), descending=True).tolist()
        ranked_documents = [context[idx] for idx in ranked_indices]
        scores = [relevance_scores[idx] for idx in ranked_indices]

        return ranked_documents, ranked_indices, scores

    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Reranks documents based on their similarity to the query using TF-IDF and a specified distance metric.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :param distance_metric: The distance metric to use for similarity calculation ('cosine', 'euclidean', 'manhattan', etc.).
        :return: A list of ranked documents, their indices, and similarity scores.
        """
        all_texts = [query] + context
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)

        # Calculate distance scores between the query (first vector) and each document
        distances = pairwise_distances(tfidf_matrix[0:1], tfidf_matrix[1:], metric=distance_metric).flatten()
        
        # Sort indices in ascending order for similarity metrics (higher is better) and descending for distance
        if distance_metric == "cosine":
            ranked_indices = np.argsort(distances)
        else:
            ranked_indices = np.argsort(distances)[::-1]

        ranked_documents = [context[idx] for idx in ranked_indices]
        scores = [distances[idx] for idx in ranked_indices]

        return ranked_documents, ranked_indices, scores

    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        """
        Combines TF-IDF and cross-encoder scores for hybrid reranking.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :param tfidf_weight: Weight for the TF-IDF score in the combined ranking.
        :param cross_encoder_weight: Weight for the cross-encoder score in the combined ranking.
        :return: A list of ranked documents, indices, and combined scores.
        """
        tfidf_documents, tfidf_indices, tfidf_scores = self.tfidf_rerank(query, context, distance_metric)
        cross_encoder_docs, _, cross_encoder_scores = self.cross_encoder_rerank(query, tfidf_documents)

        combined_scores = []
        for i, doc in enumerate(cross_encoder_docs):
            tfidf_score = tfidf_scores[tfidf_indices[i]]
            cross_encoder_score = cross_encoder_scores[i]
            combined_score = tfidf_weight * tfidf_score + (1-tfidf_weight) * cross_encoder_score
            combined_scores.append((doc, tfidf_indices[i], combined_score))

        combined_scores = sorted(combined_scores, key=lambda x: x[2], reverse=True)

        ranked_documents = [doc for doc, _, _ in combined_scores]
        ranked_indices = [idx for _, idx, _ in combined_scores]
        scores = [score for _, _, score in combined_scores]

        return ranked_documents, ranked_indices, scores

    def sequential_rerank(self, query, context, distance_metric="cosine", tfidf_top_k=5):
        """
        TF-IDF first, then cross-encoder scores for hybrid reranking.

        :param query: A string containing the query.
        :param context: A list of strings, each representing a document.
        :param tfidf_weight: Weight for the TF-IDF score in the combined ranking.
        :param tfidf_top_k: The number of top documents to select from TF-IDF reranking before cross-encoder reranking.
        :return: A list of ranked documents, indices, and combined scores.
        """
        tfidf_documents, tfidf_indices, tfidf_scores = self.tfidf_rerank(query, context, distance_metric)

        # Select the top k documents from TF-IDF reranking
        top_tfidf_docs = tfidf_documents[:tfidf_top_k]
        top_tfidf_indices = tfidf_indices[:tfidf_top_k]
        top_tfidf_scores = tfidf_scores[:tfidf_top_k]

        # Perform the cross-encoder reranking
        cross_encoder_docs, _, cross_encoder_scores = self.cross_encoder_rerank(query, top_tfidf_docs)

        reranked_results = [
            (doc, top_tfidf_indices[i], cross_encoder_scores[i]) 
            for i, doc in enumerate(cross_encoder_docs)
        ]

        reranked_results = sorted(reranked_results, key=lambda x: x[2], reverse=True)

        ranked_documents = [doc for doc, _, _ in reranked_results]
        ranked_indices = [idx for _, idx, _ in reranked_results]
        scores = [score for _, _, score in reranked_results]

        return ranked_documents, ranked_indices, scores

    def tfidf_corpus_rerank(self, query, distance_metric="cosine"):
        """
        Reranks documents from the full chunked corpus based on their similarity to the query using TF-IDF and a specified distance metric.

        :param query: A string containing the query.
        :param distance_metric: The distance metric to use for similarity calculation ('cosine', 'euclidean', 'manhattan', etc.).
        :return: A list of ranked documents, their indices, and similarity scores.
        """
        query_vector = self.vectorizer.transform([query])

        # Calculate distance scores between the query and the corpus
        distances = pairwise_distances(query_vector, self.tfidf_corpus_matrix, metric=distance_metric).flatten()
        
        # Sort indices in ascending order for similarity metrics (higher is better) and descending for distance
        if distance_metric == "cosine":
            ranked_indices = np.argsort(distances)
        else:
            ranked_indices = np.argsort(distances)[::-1]

        ranked_documents = [self.corpus[idx] for idx in ranked_indices]
        scores = [distances[idx] for idx in ranked_indices]

        return ranked_documents, ranked_indices, scores
    

if __name__ == "__main__":

    from sentence_transformers import SentenceTransformer
    import faiss
    from indexing import FaissIndex
    from search import FaissSearch

    # This is an example from SQuAD dataset. 
    # https://rajpurkar.github.io/SQuAD-explorer/
    # I also injected FALSE information!!!

    context = [
        "Vince Pulido was the first person to walk on the moon during the Apollo 11 mission in 1969.",
        "The Apollo 11 mission was a significant event in the history of space exploration.",
        "Kate Hornbeck followed Vince Pulido on the moon, making her the second person to walk on the moon.",
        "The Apollo program was designed to land humans on the moon and bring them safely back to Earth.",
        "Oxygen is a chemical element with symbol O and atomic number 20.", 
        "It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements.", 
        "By mass, oxygen is the third-most abundant element in the universe, after hydrogen and helium.", 
        ]
    
    reranker = Reranker(corpus=context, type="tfidf_corpus")

    # Embed the documents. USE THE embedding.py to implement you system.
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedding_model.encode(context)
    faiss_index_bf = FaissIndex(index_type='brute_force', nlist=50)
    faiss_index_bf.add_embeddings(doc_embeddings, metadata=context)
    faiss_index_bf = FaissSearch(faiss_index_bf, metric='euclidean')


    print("\n\n")
    # FIRST EXAMPLE
    query = "Who followed Vince Pulido to walk on the moon?"
    print("QUERY:", query)
    query_embedding = embedding_model.encode([query])
    distances_ivf, indices_ivf, metadata_ivf = faiss_index_bf.search(query_embedding, k=5)
    print("NEAREST NEIGHBORS RESULTS:")
    for i in range(5):
        print(f"Neighbor {i+1}: Index {indices_ivf[0][i]}, Distance {distances_ivf[0][i]}, Documents: {metadata_ivf[i]}")


    print("RERANKED RESULTS:")
    ranked_documents, ranked_indices, scores = reranker.rerank(query, context=metadata_ivf)
    for i in range(5):
        print(f"Rerank Document {i+1}: Scores {scores[i]}, Documents: {ranked_documents[i]}")




    print("\n\n")
    # SECOND EXAMPLE
    query = "What is the most plentiful element?"
    print("QUERY:", query)
    query_embedding = embedding_model.encode([query])
    distances_ivf, indices_ivf, metadata_ivf = faiss_index_bf.search(query_embedding, k=5)
    print("NEAREST NEIGHBORS RESULTS:")
    for i in range(5):
        print(f"Neighbor {i+1}: Index {indices_ivf[0][i]}, Distance {distances_ivf[0][i]}, Documents: {metadata_ivf[i]}")


    print("RERANKED RESULTS:")
    ranked_documents, ranked_indices, scores = reranker.rerank(query, context=metadata_ivf)
    for i in range(5):
        print(f"Reranked Document {i+1}: Scores {scores[i]}, Documents: {ranked_documents[i]}")

    