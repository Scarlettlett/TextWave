"""This file contains a class to read in text document and chunk the document"""
import os
import numpy as np
import nltk
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.indexing import FaissIndex
from modules.retrieval.search import FaissSearch
from modules.retrieval.reranker import Reranker
from modules.generator.question_answering import QA_Generator

class Pipeline:
    def __init__(self, index_type='HNSW', rerank_type="hybrid", temperature=0.6, generator_model="mistral-large-latest", **kwargs):

        # Get params
        self.metric = kwargs.get('metric', 'euclidean')
        self.p = kwargs.get('p', 3)
        self.embedding_model_name = kwargs.get('embedding_model_name', 'all-MiniLM-L6-v2')
        self.temperature = temperature
        self.generator_model = generator_model

        self.chunking_strategy = None
        self.fixed_length = None
        self.overlap_size = None

        # Initialize preprocessing, embedding, indexing, reranking, and generating
        self.processor = DocumentProcessing()
        self.embedder = Embedding(self.embedding_model_name)

        self.index_type = index_type
        self.indexer = FaissIndex(self.index_type)

        self.searcher = None

        self.corpus = None
        self.rerank_type = rerank_type
        self.reranker = None
        # self.reranker = Reranker(type=self.rerank_type, corpus=self.corpus, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.generator = QA_Generator(api_key = os.environ["MISTRAL_API_KEY"], temperature=self.temperature, generator_model=self.generator_model)
 

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

    def load_index(self, faiss_path: str, metadata_path: str):
        self.indexer.metadata = None
        self.indexer.load(faiss_path=faiss_path, metadata_path=metadata_path)

        self.corpus = self.indexer.metadata
        # Init searcher with self.index not None
        self.searcher = FaissSearch(self.indexer, self.metric, self.p)

        # Init reranker with self.corpus not None
        self.reranker = Reranker(type=self.rerank_type, corpus=self.corpus, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def index_reporting(self):
        vector_number = len(self.indexer.metadata)

        print("metadata:", self.indexer.metadata)
        print("\nchunking_strategy:", self.chunking_strategy)
        print("\nfixed_length:", self.fixed_length)
        print("\noverlap_size:", self.overlap_size)
        print("\nnumber_of_vectors:", vector_number)
        print("\nvector_dimension:", self.indexer.vector_dimension)

        return self.chunking_strategy, self.fixed_length, self.overlap_size, vector_number
    
    def __encode(self, query):
        # Preprocess the query
        query = query.strip()
        query_reshaped = nltk.sent_tokenize(query)
        # Embed the query
        return self.embedder.encode(query_reshaped)

    def search_rerank(self, query, k, type="hybrid", reporting=True):
        # Embed the query
        query_embedding = self.__encode(query)
        
        # Search the query
        distances_ivf, indices_ivf, metadata_ivf = self.searcher.search(query_embedding, k)

        if reporting:
            # Reporting the search results
            print("QUERY:", query)
            print("\nNEAREST NEIGHBORS RESULTS:")
            for i in range(k):
                print(f"Neighbor {i+1}: Index {indices_ivf[0][i]}, Distance {distances_ivf[0][i]}, Documents: {metadata_ivf[i]}")

        # Initiate reranker from retrival module
        self.reranker = Reranker(type=type, corpus=self.corpus, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
        ranked_documents, ranked_indices, scores = self.reranker.rerank(query, context=metadata_ivf, distance_metric="cosine")

        if reporting:
            # Reporting the reranked results
            print("\nRERANKED RESULTS:")
            for i in range(k):
                print(f"Rerank Document {i+1}: Scores {scores[i]}, Documents: {ranked_documents[i]}")
        
        return ranked_documents, ranked_indices, scores
    
    def search_neighbors(self, query_embedding, k=40, reporting=True):
        query_embedding_vector = self.__encode(query_embedding)
        # Search the query
        distances_ivf, indices_ivf, metadata_ivf = self.searcher.search(query_embedding_vector, k)
        
        if reporting:
            # Reporting the search results
            print("QUERY:", query_embedding)
            print("\nNEAREST NEIGHBORS RESULTS:")
            for i in range(k):
                print(f"Neighbor {i+1}: Index {indices_ivf[0][i]}, Distance {distances_ivf[0][i]}, Documents: {metadata_ivf[i]}")
        
        return metadata_ivf
    
    def generate_answer(self, query, context, rerank=True, n=10, reporting=True):
        import time
        from mistralai.models import SDKError

        # Initiate reranker from retrival module
        self.reranker = Reranker(type=self.rerank_type, corpus=self.corpus, cross_encoder_model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')

        if rerank:
            retrived_documents,_,_ = self.reranker.rerank(query=query, context=context, distance_metric="cosine")
            
        else:
            retrived_documents = context

        retrived_documents = retrived_documents[:n] if len(retrived_documents) > n else retrived_documents

        # Use QA_generator to generate an answer
        try:
            generated_answer = self.generator.generate_answer(query, retrived_documents)
        except SDKError as e:
            if e.status_code == 429:  # Rate limit error
                print("Rate limit exceeded. Waiting for 10 seconds before retrying...")
                time.sleep(10)  # Wait before retrying
                generated_answer = self.generator.generate_answer(query, retrived_documents)

        if reporting:
            # Reporting the search results
            print("QUERY:", query)
            print("\nNEAREST NEIGHBORS RESULTS:")
            for i in range(len(retrived_documents)):
                print(f"Neighbor {i+1}: Documents: {retrived_documents[i]}")
            print("\nGENERATED ANSWER:", generated_answer)

        return generated_answer
    
    def add_to_corpus(self, document_id, document_text):
        """Add a single document to the corpus, embed it, and update the Faiss index."""
        if document_text in self.corpus:
            raise ValueError(f"Document '{document_id}' already exists in the corpus.")

        # 1. Chunk the document based on the chunking strategy
        if self.chunking_strategy == 'sentence' and self.overlap_size:
            chunks = self.processor.sentence_chunking_from_text(document_text, overlap_size=self.overlap_size)
        elif self.chunking_strategy == 'fixed-length' and self.fixed_length:
            chunks = self.processor.fixed_length_chunking_from_text(document_text, fixed_length=self.fixed_length, overlap_size=self.overlap_size)
        else:
            raise ValueError("Chunking strategy is invalid. Please preprocess the corpus first.")

        # 2. Embed and index each chunk
        for chunk in chunks:
            sentence_embedding = self.embedder.encode(chunk)
            sentence_embedding = np.array(sentence_embedding).reshape(1, -1)

            # Add embedding to Faiss index and metadata
            self.indexer.add_embeddings(sentence_embedding, metadata=chunk)

        # 3. Update the corpus with the new chunks
        self.corpus.extend(chunks)

        # Log the addition
        print(f"Document '{document_id}' added to the corpus successfully.")

    

if __name__ == "__main__":
    corpus_dir = "storage\\sample_corpus"
    sample_faiss_path = "storage\\index\\faiss_index.bin"
    sample_metadata_path = "storage\\index\\metadata.pkl"
    pipeline = Pipeline(index_type='IVF')
    # pipeline.preprocess_corpus(paragraph_dir, chunking_strategy='fixed-length', fixed_length=50, overlap_size=3)
    pipeline.preprocess_corpus(corpus_dir, chunking_strategy='sentence', fixed_length=60, overlap_size=1)
    # pipeline.load_index(sample_faiss_path, sample_metadata_path)
    # pipeline.index_reporting()
    # pipeline.indexer.save(sample_faiss_path,sample_metadata_path)
    query = "When did Lincoln begin his political career?"
    retrived_docs = pipeline.search_neighbors(query, k=10, reporting=True)

    # Use QA generator to generate answer
    pipeline.generate_answer(query, retrived_docs, rerank=True, reporting=True)
