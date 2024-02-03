import logging
import logging.config
import dotenv
import os
from rich.logging import RichHandler

# Load environment variables
dotenv.load_dotenv()

# Define the configuration file path based on the environment
config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
rich_handler = RichHandler()
# Replace the first handler, assuming it's the console handler
logging.getLogger().handlers[0] = rich_handler

import sys
import glob
import random

# Append the directory above 'scripts' to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the WhatsAppChatLoader class from where it is defined
from ai.lib.loaders.whatsapp_loader import WhatsAppChatLoader 
from ai.lib import lib_model, lib_doc_vectors
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter

def main():
    path = sys.argv[1]

    # Initialize the library with environment variables
    lib_model.init(os.getenv("SMART_OPENAI_MODEL"), os.getenv("FAST_OPENAI_MODEL"), os.getenv("OPENAI_API_KEY"), os.getenv("PGVECTOR_CONNECTION_STRING"), os.getenv("RECORDMANAGER_CONNECTION_STRING"), temp=os.getenv("OPENAI_TEMPERATURE"))

    txt_files = list(glob.glob(os.path.join(path, "*.txt")))

    random.shuffle(txt_files)
    total_files = len(txt_files)
    docs = []

    for i, file_path in enumerate(txt_files):
        print(f"Processing file {i+1}/{total_files}: {file_path}")
        loader = WhatsAppChatLoader(file_path)
        docs.extend(loader.load())
        print(f"Length of docs: {len(docs)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(2000), chunk_overlap=200, add_start_index=True)
    all_docs = text_splitter.split_documents(docs)
    print(f"Length of all_docs: {len(all_docs)}")
    # for doc in loader.load():
    #     text = doc.page_content
    #     metadata = doc.metadata
    #     print(f"\n\nAdding document title:{metadata['title']} {len(text)} first 50 chars: {text[:50]}")
    #     # print(doc.page_content)
    #     # Assuming lib_doc_vectors.bulk_add_docs is the method to add documents to your storage or processing pipeline
    lib_doc_vectors.bulk_add_docs(all_docs)

if __name__ == "__main__":
    main()