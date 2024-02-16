import argparse
import uuid
import os
from halo import Halo
from prompt_toolkit import prompt
from prompt_toolkit.key_binding import KeyBindings
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory

from langchain_core.agents import AgentAction, AgentFinish
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableLambda

from prompt_toolkit.history import FileHistory
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.vectorstores.pgvector import PGVector
from langchain_core.documents import Document
from langchain.agents import AgentExecutor

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.callbacks.manager import Callbacks
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.tools.retriever import create_retriever_tool



from langchain.agents import OpenAIFunctionsAgent



from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter
import datetime
import dateutil
import logging

from . import prompts


lmd = None
db = None
llm = None
retriever = None
chat_history = None
memory = None
conversation_file = None
tw_retriever = None
main_agent = None

class ExpertAgent(OpenAIFunctionsAgent):
    def plan(
            self,
            intermediate_steps: List[Tuple[AgentAction, str]],
            callbacks: Callbacks = None,
            **kwargs: Any,
        ) -> Union[AgentAction, AgentFinish]:
            """Given input, decided what to do.

            Args:
                intermediate_steps: Steps the LLM has taken to date,
                    along with observations
                callbacks: Callbacks to run.
                **kwargs: User inputs.

            Returns:
                Action specifying what tool to use.
            """

            logger = logging.getLogger(__name__)

            logger.debug(f"EXPERTAGENT: Planning")
            res = super(self).plan(intermediate_steps, callbacks, **kwargs)

            
            logger.debug(f"PLAN: {res}")

            return res

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import BasePromptTemplate
from typing import Any, List, Optional, Sequence, Tuple, Union
