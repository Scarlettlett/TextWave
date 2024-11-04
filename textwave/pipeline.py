"""This file contains a class to read in text document and chunk the document"""
import os
import numpy as np
import nltk
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.indexing import FaissIndex
from modules.retrieval.search import FaissSearch
from modules.retrieval.reranker import Reranker

class Pipeline:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', index_type='brute_force', **kwargs):
        self.embedding_model_name = embedding_model_name
        self.index_type = index_type
        # Initialize preprocessing, embedding, and indexing
        self.processor = DocumentProcessing()
        self.embedder = Embedding(self.embedding_model_name)
        self.indexer = FaissIndex(self.index_type)
        self.searcher = None

        # Get params for initiating searching in method preprocess_corpus
        self.metric = kwargs.get('metric', 'euclidean')
        self.p = kwargs.get('p', 3)
        
        self.chunking_strategy = None
        self.fixed_length = None
        self.overlap_size = None
        self.corpus = None

    def preprocess_corpus(self, corpus_directory, chunking_strategy='sentence', fixed_length=None, overlap_size=2):
        """Chunk each file in the corpus directory, embed each chunk, and store chunks"""
        self.chunking_strategy = chunking_strategy
        self.fixed_length = fixed_length
        self.overlap_size = overlap_size
        document_chunks = []

        #1. Chunk each file
        for doc_name in os.listdir(corpus_directory):
            doc_path = os.path.join(corpus_directory, doc_name)
            # If specified to use overlap chunking strategy
            if chunking_strategy == 'sentence' and overlap_size:
                chunks = self.processor.sentence_chunking(doc_path, overlap_size=overlap_size)
            # If specified to use fixed-length chunking strategy
            elif chunking_strategy == 'fixed-length' and fixed_length:
                chunks = self.processor.fixed_length_chunking(doc_path, fixed_length=fixed_length, overlap_size=overlap_size)
            else:
                return f"Input parameters are invalid. Chunking_strategy can be either sentence or fixed-length."
        
        #2. Embed each chunk
            for chunk in chunks:
                sentence_embedding = self.embedder.encode(chunk)
                sentence_embedding = np.array(sentence_embedding).reshape(1, -1)

        #3. Add embedding to index and store the chunk as metadata
                self.indexer.add_embeddings(sentence_embedding, metadata=chunk)

            document_chunks.extend(chunks) # Store chunks for each document

        #4. Initiate searching FaissSearch after index created in add_embeddings
        self.searcher = FaissSearch(self.indexer, self.metric, self.p)

        #5. Update self.corpus to a list of all chunked documents
        self.corpus = document_chunks
    
    def index_reporting(self):
        vector_number = len(self.indexer.metadata)

        print("metadata:", self.indexer.metadata)
        print("\nchunking_strategy:", self.chunking_strategy)
        print("\nfixed_length:", self.fixed_length)
        print("\noverlap_size:", self.overlap_size)
        print("\nnumber_of_vectors:", vector_number)
        print("\nvector_dimension:", self.indexer.vector_dimension)

        return self.chunking_strategy, self.fixed_length, self.overlap_size, vector_number
    
    def search_rerank(self, query, k, type="hybrid", reporting=True):
        # Preprocess the query
        query = query.strip()
        query_reshaped = nltk.sent_tokenize(query)
        # Embed the query
        query_embedding = self.embedder.encode(query_reshaped)
        
        # Search the query
        distances_ivf, indices_ivf, metadata_ivf = self.searcher.search(query_embedding, k)

        if reporting:
            # Reporting the search results
            print("QUERY:", query)
            print("\nNEAREST NEIGHBORS RESULTS:")
            for i in range(5):
                print(f"Neighbor {i+1}: Index {indices_ivf[0][i]}, Distance {distances_ivf[0][i]}, Documents: {metadata_ivf[i]}")

        # Initiate reranker from retrival module
        self.reranker = Reranker(type=type, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        ranked_documents, ranked_indices, scores = self.reranker.rerank(query, context=metadata_ivf, corpus=self.corpus)

        if reporting:
            # Reporting the reranked results
            print("\nRERANKED RESULTS:")
            for i in range(k):
                print(f"Rerank Document {i+1}: Scores {scores[i]}, Documents: {ranked_documents[i]}")
        
        return ranked_documents, ranked_indices, scores

    

if __name__ == "__main__":
    corpus_dir = "storage\\sample_corpus"
    sample_faiss_path = "storage\\index\\faiss_index.bin"
    sample_metadata_path = "storage\\index\\metadata.pkl"
    pipeline = Pipeline(embedding_model_name='all-MiniLM-L6-v2', index_type='brute_force')
    # pipeline.preprocess_corpus(paragraph_dir, chunking_strategy='fixed-length', fixed_length=50, overlap_size=3)
    pipeline.preprocess_corpus(corpus_dir, chunking_strategy='sentence', fixed_length=60, overlap_size=1)
    # pipeline.index_reporting()
    # pipeline.indexer.save(sample_faiss_path,sample_metadata_path)
    query = "When did Lincoln begin his political career?"
    pipeline.search_rerank(query, k=5, type="sequential")
