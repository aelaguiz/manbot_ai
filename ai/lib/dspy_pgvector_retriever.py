import dspy

from .lib_docdb import get_docdb
import logging
from typing import Optional, List, Union
from dsp.utils import dotdict

class PGVectorRM(dspy.Retrieve):
    def __init__(
        self,
        k: int = 3,
    ):

        self.docdb = get_docdb()
        self.retriever = self.docdb.as_retriever(search_kwargs={'k': k})
        super().__init__(k=k)


    def forward(
        self, query_or_queries: Union[str, List[str]], k: Optional[int] = None
    ) -> dspy.Prediction:
        logger = logging.getLogger(__name__)
        # logger.debug(f"Getting documents for query: {query_or_queries}")
        res = self.retriever.get_relevant_documents(query_or_queries)

        # logger.debug(f"Retrieved documents: {len(res)}")
        # for doc in res:
        #     logger.debug(f"Document: {doc.page_content}")

        return dspy.Prediction(passages=[doc.page_content for doc in res])