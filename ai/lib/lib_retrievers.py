import logging
from langchain_core.callbacks.manager import AsyncCallbackManagerForRetrieverRun
from pydantic import BaseModel, Field
from typing import Any, Coroutine, List
from datetime import datetime
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List




document_metadata_attributes = [
    AttributeInfo(name="type", type="string", description="Type of the document"),
    AttributeInfo(name="source", type="string", description="Source of the document"),
    AttributeInfo(name="created_at", type="int", description="Creation timestamp of the document"),
    AttributeInfo(name="name", type="string", description="Name of the document")
]


class RetrieverWrapper(VectorStoreRetriever):
    def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun) -> Coroutine[Any, Any, List[Document]]:

        logger = logging.getLogger(__name__)
        # logger.debug(f"RetrieverWrapper: _aget_relevant_documents: {query}")
        res = super()._aget_relevant_documents(query, run_manager=run_manager)
        # logger.debug(f"RetrieverWrapper: _aget_relevant_documents: {res}")
        return res
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        logger = logging.getLogger(__name__)
        # logger.debug(f"RetrieverWrapper: _get_relevant_documents: {query}")
        res = super()._get_relevant_documents(query, run_manager=run_manager)

        # logger.debug(f"RetrieverWrapper: _get_relevant_documents: {res}")
        return res



def get_retriever(vectorstore, k, source_filter=None, type_filter=None):
    skw = get_search_kwargs(k, source_filter, type_filter)

    tags = []
    tags.extend(vectorstore._get_retriever_tags())

    return RetrieverWrapper(vectorstore=vectorstore, search_type="similarity", search_kwargs=skw, tags=tags)

def get_self_query(llm, vectorstore, doc_content_description, k, source_filter=None, type_filter=None):
    logger = logging.getLogger(__name__)
    skw = get_search_kwargs(k, source_filter, type_filter)
    logger.debug(f"Initializing kwargs", skw)
    sq = SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        doc_content_description,
        document_metadata_attributes,
        structured_query_translator=PGVectorTranslator(),
        search_kwargs=skw)


    logger.debug(f"SelfQueryRetriever: {sq.search_kwargs}")

    return sq


def get_search_kwargs(k, source_filter, type_filter):
    search_kwargs = {
        "k": k
    }

    if source_filter and type_filter:
        raise Exception("Not Implemented: Cannot filter by both source and type")

    if source_filter:
        ## This syntax was neeeded for chroma, but I don't think needed for pgvector
        # search_kwargs["filter"] = {"source": {"$eq": source_filter}}
        search_kwargs["filter"] = {"source": source_filter}
    if type_filter:
        search_kwargs["filter"] = {"type": type_filter}

    return search_kwargs



from typing import TYPE_CHECKING, Tuple, Union

from langchain.chains.query_constructor.ir import (
    Comparator,
    Comparison,
    Operation,
    Operator,
    StructuredQuery,
    Visitor,
)


class PGVectorTranslator(Visitor):
    """Translate the internal query language elements to valid filters."""

    allowed_operators = [Operator.AND, Operator.OR, Operator.NOT]
    """Subset of allowed logical operators."""

    allowed_comparators = [
        Comparator.EQ,
        Comparator.GT,
        Comparator.GTE,
        Comparator.LT,
        Comparator.LTE,
    ]

    COMPARATOR_MAP = {
        Comparator.EQ: "==",
        Comparator.GT: ">",
        Comparator.GTE: ">=",
        Comparator.LT: "<",
        Comparator.LTE: "<=",
    }
    OPERATOR_MAP = {Operator.AND: "AND", Operator.OR: "OR", Operator.NOT: "NOT"}

    def _format_func(self, func: Union[Operator, Comparator]) -> str:
        self._validate_func(func)
        if isinstance(func, Operator):
            value = self.OPERATOR_MAP[func.value]  # type: ignore
        elif isinstance(func, Comparator):
            value = self.COMPARATOR_MAP[func.value]  # type: ignore
        return f"{value}"

    def visit_operation(self, operation: Operation):
        print(f"PGVectorTranslator visit_operation: {operation}")

        args = [arg.accept(self) for arg in operation.arguments]
        return None

    def visit_comparison(self, comparison: Comparison):
        print(f"PGVectorTranslator visit_comparison: {comparison}")
        return None

    def visit_structured_query(
        self, structured_query: StructuredQuery
    ) -> Tuple[str, dict]:
        # print("PGVectorTranslator visit_structured_query", structured_query, structured_query.query, structured_query.filter)
        if structured_query.filter is None:
            kwargs = {}
        else:
            kwargs = {"predicates": structured_query.filter.accept(self)}
        # print("PGVectorTranslator visit_structured_query", structured_query.query, "kwargs", kwargs)
        return structured_query.query, kwargs
