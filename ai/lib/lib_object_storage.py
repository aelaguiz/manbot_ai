# lib_object_storage.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv


mongo_uri = f"mongodb+srv://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASS')}@{os.getenv('MONGO_CLUSTER_URL')}/{os.getenv('MONGO_DB_NAME')}?retryWrites=true&w=majority"

# Establish MongoDB connection
client = MongoClient(mongo_uri)
db = client[os.getenv("MONGO_DB_NAME")]

def get_collection(obj_type):
    """
    Get or create a collection for a specific object type.
    """
    # Normalize the collection name, e.g., converting it to lowercase
    collection_name = obj_type.lower()
    return db[collection_name]

def generate_combined_id(obj_type, obj_id):
    """
    Generate a combined ID in the format "{type}_{id}".
    """
    return f"{obj_type}_{obj_id}"

def check_nonexistent_ids(obj_type, id_list):
    """
    Checks and returns the list of IDs that do not exist in the database for a given object type.

    :param obj_type: The type of object to query (e.g., 'video', 'audio').
    :param id_list: A list of IDs to check for existence in the database.
    :return: A list of IDs that do not exist in the database.
    """
    # Get the appropriate collection based on obj_type
    collection = get_collection(obj_type)

    combined_id_list = [generate_combined_id(obj_type, obj_id) for obj_id in id_list]

    # Prepare the query to find all documents where the ID is in the combined_id_list
    query = {"_id": {"$in": combined_id_list}}
    found_documents = collection.find(query, {"_id": 1})  # We're only interested in the '_id' field

    # Extract the set of found IDs from the query results
    found_ids = {doc['_id'] for doc in found_documents}

    # Determine which of the provided IDs were not found in the database
    nonexistent_ids = set(id_list) - found_ids

    return list(nonexistent_ids)


def bulk_store_documents(obj_type, key, documents):
    """
    Bulk store a list of documents in the collection corresponding to the object type.
    The key is used to extract the object ID from each document's metadata.

    Parameters:
    documents (list): A list of documents, each with page_content and metadata.
    obj_type (str): The object type determining which collection to store documents in.
    key (str): The key in the metadata dict to use as the object ID.
    """
    # Access the collection for the given object type
    collection = get_collection(obj_type)
    bulk_docs = []

    # Iterate over the documents to prepare them for bulk insertion
    for doc in documents:
        obj_id = doc.metadata.get(key)  # Extract the obj_id using the specified key from metadata
        if obj_id is None:
            raise ValueError(f"Key '{key}' not found in metadata for one of the documents")

        combined_id = generate_combined_id(obj_type, obj_id)  # Create a unique ID for the document

        # Create the document to be inserted
        document = {
            "_id": combined_id,
            "type": obj_type,
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        bulk_docs.append(document)

    # Perform the bulk insert operation
    if bulk_docs:  # Ensure there's something to insert
        collection.insert_many(bulk_docs)

def store_object(obj_type, obj_id, page_content, metadata):
    """
    Store an object in the specified collection based on object type.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    document = {
        "_id": combined_id,
        "type": obj_type,  # Keeping type for easier retrieval in a single collection scenario
        "page_content": page_content,
        "metadata": metadata
    }
    collection = get_collection(obj_type)
    collection.insert_one(document)

def retrieve_object(obj_type, obj_id):
    """
    Retrieve an object by its type and ID from the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    return collection.find_one({"_id": combined_id})

def get_all_objects_of_type(obj_type):
    """
    Retrieve all objects of a specific type from the corresponding collection.
    """
    collection = get_collection(obj_type)
    return list(collection.find({}))

def check_object_exists(obj_type, obj_id):
    """
    Check if an object exists in the corresponding collection by its type and ID.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    return collection.count_documents({"_id": combined_id}) > 0

def update_page_content(obj_type, obj_id, new_page_content):
    """
    Update the page_content of an object in the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    collection.update_one({"_id": combined_id}, {"$set": {"page_content": new_page_content}})

def update_metadata(obj_type, obj_id, new_metadata):
    """
    Update the metadata of an object in the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    collection.update_one({"_id": combined_id}, {"$set": {"metadata": new_metadata}})

def delete_object(obj_type, obj_id):
    """
    Delete an object by its type and ID from the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    collection.delete_one({"_id": combined_id})