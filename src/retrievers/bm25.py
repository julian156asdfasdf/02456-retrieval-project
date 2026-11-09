from scipy.sparse import lil_matrix
from src.retrievers.tf_idf import TFIDF
from src.data_utils.document import Document
import numpy as np

class BM25(TFIDF):
    def __init__(self, documents: list[dict] = None, index_path: str = None, k1: float = 1.5, b: float = 0.75) -> None:
        super(BM25, self).__init__(documents, index_path)
        self.k1 = k1
        self.b = b
        self.bm25_matrix = self.GetDocumentBM25Vectors()
    
    def GetDocumentLengths(self):
        document_lengths = {}
        average_document_length = 0
        for document in self.index.GetDocuments():
            length = len(self.PreprocessText(document.GetText()))
            document_lengths[document] = length
            average_document_length += length
        return document_lengths, average_document_length/len(self.index.GetDocuments())
    
    # TODO: Add methods for computing self.bm25_matrix
    def GetInverseDocumentFrequencies(self) -> dict[str, float]:
        """
        Returns a dict with terms and their corresponding inverse document frequencies.
        """
        idf = self.GetDocumentFrequencies()
        N = len(self.index.GetDocuments())

        for word in self.corpus_vocabulary:
            idf[word] = np.log((N+1)/(idf[word] + 0.5))


        return idf

    def GetDocumentBM25Vectors(self):
        """
        Returns a n_documents x n_terms matrix whose element (i, j) is the td-idf score of the j'th term in the i'th document.
        """
        # Create a mapping from terms to indices
        self.term_to_index = {term: idx for idx, term in enumerate(self.corpus_vocabulary)}

        document_lengths, average_document_length = self.GetDocumentLengths()

        # Initialize a sparse matrix
        n_documents = len(self.index.GetDocuments())
        n_terms = len(self.corpus_vocabulary)
        bm25_matrix = lil_matrix((n_documents, n_terms), dtype=np.float32)
        from tqdm import tqdm

        for i, doc in tqdm(enumerate(self.index.GetDocuments())):
            doc_indexes = [self.term_to_index[word] for word in self.GetDocumentVocabulary(doc)]

            for doc_index in doc_indexes:
                bm25_matrix[i, doc_index] += 1

            for term in set(self.GetDocumentVocabulary(doc)):
                fij = bm25_matrix[i, self.term_to_index[term]]
                bm25_matrix[i, self.term_to_index[term]] = (fij * (self.k1 + 1)) / (fij + self.k1*(1 - self.b + self.b * (document_lengths[doc]/average_document_length)))

        return bm25_matrix

    
    def CalculateScores(self, queries: list[str]):
        """Calculate scores for a query
        
        Args:
            query (str): The query to calculate scores for
        
        Returns:
            np.array: An array of scores for each document
        """
        query_vector = self.QueryToVector(queries)
        scores = query_vector.dot(self.bm25_matrix.T).toarray()
        return scores