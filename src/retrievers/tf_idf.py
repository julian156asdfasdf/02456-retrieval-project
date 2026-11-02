from src.data_utils.document import Document
from src.retrievers.base_retriever import Retriever
import numpy as np
from scipy.sparse import lil_matrix
import re


class TFIDF(Retriever):
    def __init__(self, documents: list[dict] = None, index_path: str = None) -> None:
        super(TFIDF, self).__init__(documents, index_path)
        self.corpus_vocabulary = self.GetCorpusVocabulary()
        self.idf = self.GetInverseDocumentFrequencies()

        # Only run if we are in TFIDF class
        if type(self) == TFIDF:
            self.tfidf_matrix = self.GetDocumentsTFIDFVectors()

    def PreprocessText(self, text: str) -> list[str]:
        """
        Remove unwanted charatcters from text and lowercase
        """
        text = text.lower()

        cleaned = "".join(char for char in text if char.isalnum() or char.isspace())

        return cleaned.split()        
    def GetQueryVocabulary(self, query: str) -> list[str]:
        """
        Returns all unique words in a query
        """
        return self.PreprocessText(query)

    def GetDocumentVocabulary(self, document: Document) -> list[str]:
        """
        Returns all unique words in a document
        """
        return self.PreprocessText(document.GetText())
    
    def GetCorpusVocabulary(self) -> set[str]:
        """
        Returns all unique words in ALL documents.
        """
        # Use map function for more efficient processing
        processed_texts = map(lambda doc: set(self.PreprocessText(doc.GetText())), self.index.GetDocuments())
        corpus_vocabulary = set().union(*processed_texts)
        return corpus_vocabulary

    def GetDocumentFrequencies(self) -> dict[str, int]:
        """
        Returns a dict with terms and counts of how many documents the term occurs in (counted once per document).
        """
        df = {}
        df = {word: 0 for word in self.corpus_vocabulary}

        for doc in self.index.GetDocuments():
            text = set(self.GetDocumentVocabulary(doc))
            for word in set(self.GetDocumentVocabulary(doc)):   # set -> once per document
                df[word] += 1

        return df
       
    
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        idf = self.GetDocumentFrequencies()
        N = len(self.index.GetDocuments())

        for word in self.corpus_vocabulary:
            idf[word] = np.log(N/idf[word])


        return idf
    
    
    def GetDocumentTermCounts(self, document: Document):
        """
        Returns a dict on the form {term: num_occurences} for an input Document.
        """
        term_counts = {word: 0 for word in self.corpus_vocabulary}
        for word in self.GetDocumentVocabulary(document):
            term_counts[word] += 1


        return term_counts

    def GetDocumentsTFIDFVectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        tfidf_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)
        from tqdm import tqdm

        for i, doc in tqdm(enumerate(self.index.GetDocuments())):
            doc_indexes = [self.term_to_index[word] for word in self.GetDocumentVocabulary(doc)]

            for doc_index in doc_indexes:
                tfidf_matrix[i, doc_index] += 1

            for term in set(self.GetDocumentVocabulary(doc)):
                tfidf_matrix[i, self.term_to_index[term]] = tfidf_matrix[i, self.term_to_index[term]] * self.idf[term]






        
        return tfidf_matrix.tocsr()   # Convert to CSR format for efficient row slicing
    
    def QueryToVector(self, queries: list[str]):
        """ Converts a query or batch of queries into a sparse vector matching the vocab size with counts of how many times the term occurs.
        
        Args:
            query (str): The query to convert
        
        Returns:
            csr_matrix: A sparse vector representation of the query
        """
        query_vector = lil_matrix((len(queries), len(self.corpus_vocabulary)), dtype=np.float32)

        for i, query in enumerate(queries):
            query_vocab = self.GetQueryVocabulary(query)
            query_index = [self.term_to_index[word] for word in query_vocab if word in self.term_to_index]

            for index in query_index:
                query_vector[i, index] = +1
        

        return query_vector.tocsr()

    #@time_func
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document

            num_queries x num_documents where (i, j) = ∑_i TF-IDF(q_i, D_j; D) 
        
        """
        # Convert the queries to a vector
        query_vectors = self.QueryToVector(queries) # n_queries x vocab size
        scores = query_vectors.dot(self.tfidf_matrix.T).toarray() # n_queries x n_documents
        return scores
    