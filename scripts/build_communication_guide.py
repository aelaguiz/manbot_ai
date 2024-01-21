import sys
import os

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai.lib import ai_defaults, lib_model, lib_doc_vectors, prompts, lc_logger
from ai.lib.loaders import wp_loader

import logging
import logging.config
import dotenv
import os

dotenv.load_dotenv()


# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.connection").setLevel(logging.CRITICAL)
logging.getLogger("httpcore.http11").setLevel(logging.CRITICAL)
logging.getLogger("openai._base_client").setLevel(logging.CRITICAL)


def main():
    lmd = lc_logger.LlmDebugHandler()
    comguide_path = sys.argv[1]
    # if os.path.exists(comguide_path):
    #     default_guide = False
    #     comguide = open(comguide_path, "r").read()
    # else:
    default_guide = True
    comguide = ai_defaults.comguide_default

    print("Loading model")

    lib_model.init(os.getenv("OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    print("Done loading")

    vectordb = lib_doc_vectors.get_vectordb()
    print(vectordb)

    loader = wp_loader.WPLoader('documents/innerconfidence.WordPress.2023-12-29.xml')
    docs = loader.load()
    print(len(docs))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(2000), chunk_overlap=200, add_start_index=True)
    all_docs = text_splitter.split_documents(docs)

    llm = lib_model.get_json_llm()
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompts.comguide_prompt),
        HumanMessagePromptTemplate.from_template("{input}"),
    ])
    merge_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(prompts.merge_comguides_prompt),
        HumanMessagePromptTemplate.from_template("**Guide 1:**\n{input_1}"),
        HumanMessagePromptTemplate.from_template("**Guide 2:**\n{input_2}"),
    ])

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    merge_chain = (
        merge_prompt
        | llm
        | StrOutputParser()
    )

    guidelog = open('guidelog.txt', 'w')
    for doc in docs:
        new_comguide = chain.invoke({
            "input": doc.page_content,
            "comguide": comguide
        }, config={'callbacks': [lmd]})


        guidelog.write(f"\n\n********************************************************\n")
        guidelog.write(f"Document: {doc.metadata['title']}\n")
        guidelog.write(f"Content: {doc.page_content}\n\n")
        guidelog.write(f"Old comguide: {comguide}\n\n")
        guidelog.write(f"New comguide: {new_comguide}\n\n")

        # if not default_guide:
        #     merged_comguide = merge_chain.invoke({
        #         "input_1": comguide,
        #         "input_2": new_comguide
        #     }, config={'callbacks': [lmd]})

        #     guidelog.write(f"Merged comguide: {merged_comguide}\n\n")
        #     break
        # else:
        #     merged_comguide = new_comguide

        guidelog.flush()

        # with open(comguide_path, "w") as f:
        #     f.write(merged_comguide)

        # comguide = merged_comguide
        comguide = new_comguide
        default_guide = False


if __name__ == "__main__":
    main()