import sys
import os

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import logging
import logging.config
import dotenv
import os

dotenv.load_dotenv()

# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')
persist_dir = os.getenv("LLAMA_PERSIST_DIR")

# # Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)

import html
import html2text
import llama_index

from llama_index import download_loader

if not os.path.exists(persist_dir):
    WordpressReader = download_loader("WordpressReader")
    loader = WordpressReader(url='https://members.innerconfidence.com', username='aelaguiz@gmail.com', password='UDWU yjke M0oU truC IdcV GbgZ')
    print(loader)
    documents = loader.load_data()
    print(documents)
    index = llama_index.VectorStoreIndex.from_documents(documents)
    print(index)

    print("Persist dir: ", persist_dir)
    index.storage_context.persist(persist_dir=persist_dir)
else:
    storage_context = llama_index.StorageContext.from_defaults(persist_dir=persist_dir)
    index = llama_index.load_index_from_storage(storage_context)



# from langchain.text_splitter import RecursiveCharacterTextSplitter


# from ai.lib import lib_model, lib_doc_vectors
# from ai.lib.loaders import wp_loader


# import logging
# import logging.config
# import dotenv
# import os

# dotenv.load_dotenv()



# def main():


    # lib_model.init(os.getenv("OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    # vectordb = lib_doc_vectors.get_vectordb()
    # print(vectordb)

    # loader = wp_loader.WPLoader('documents/innerconfidence.WordPress.2023-12-29.xml')
    # docs = loader.load()
    # print(len(docs))

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(2000), chunk_overlap=200, add_start_index=True)
    # all_docs = text_splitter.split_documents(docs)

    # # for doc in all_docs:
    # #     text = doc.page_content
    # #     metadata = doc.metadata
    # #     print(f"Adding document title:{metadata['title']} author:{metadata['author']} guid:{metadata['guid']} link:{metadata['url']}  start_index:{metadata['start_index']} {len(text)} first 50 chars: {text[:50]}")
    # #     print(doc.page_content)

    # #     # lib_doc_vectors.add_doc(doc, metadata['guid'])
    # print(f"Adding {len(all_docs)} documents...")
    # lib_doc_vectors.bulk_add_docs(all_docs)
    # print("Done")



# if __name__ == "__main__":
#     main()