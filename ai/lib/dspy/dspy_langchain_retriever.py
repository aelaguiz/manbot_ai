import dspy
from typing import List, Union, Optional

from .. import lib_retrievers

class LangchainRetriever(dspy.Retrieve):
    def __init__(self, vectorstore, k: int = 3, source_filter=None, type_filter=None):
        """
        Initializes the LangchainRetriever with a specific VectorStore and retrieval parameters.

        Args:
            vectorstore: The vectorstore instance to use for document retrieval.
            k (int, optional): The number of top documents to retrieve. Defaults to 3.
            source_filter: Optional filter criteria for the source of documents.
            type_filter: Optional filter criteria for the type of documents.
        """
        self.retriever = lib_retrievers.get_retriever(vectorstore, k, source_filter, type_filter)
        self.k = k
        super().__init__(k=k)

    def forward(self, query_or_queries: Union[str, List[str]], k: Optional[int] = None) -> List[dspy.Prediction]:
        """
        Search the Langchain database for the top k documents for the given query.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            k (Optional[int], optional): The number of documents to retrieve. Overrides the default if provided.

        Returns:
            List[dspy.Prediction]: A list of dspy.Prediction objects containing the retrieved documents.
        """
        k = self.k if k is None else k
        queries = [query_or_queries] if isinstance(query_or_queries, str) else query_or_queries

        results = []
        for query in queries:
            # Retrieve documents based on the query
            documents = self.retriever._get_relevant_documents(query, run_manager=None)  # run_manager is not used here
            
            # Limit the results to top k
            limited_docs = documents[:k]
            
            # Format the documents into dspy.Prediction objects
            for doc in limited_docs:
                doc_dict = self._document_to_dict(doc)
                results.append(doc_dict)

        return results

    @staticmethod
    def _document_to_dict(doc) -> dict:
        """
        Converts a Langchain document to a dictionary format.

        Args:
            doc: A document retrieved from the Langchain database.

        Returns:
            dict: A dictionary representing the document, with relevant fields.
        """
        # Convert the Langchain Document object to a dictionary format expected by dspy
        # Adjust the fields based on your requirements and document structure
        return {
            "long_text": doc.page_content,  # Use 'page_content' for the document text
            "metadata": doc.metadata  # Include any additional metadata you want to carry over
        }
