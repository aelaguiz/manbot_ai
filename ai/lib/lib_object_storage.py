# lib_object_storage.py
import os
import firebase_admin
from firebase_admin import credentials, firestore
import logging

_firestore_client = None

logger = logging.getLogger(__name__)

def init(config_path):
    """
    Initialize Firebase Admin using the provided config path and obtain Firestore client.
    """
    global _firestore_client

    if _firestore_client is not None:
        logger.error("Firebase has already been initialized.")
        raise Exception("Firebase has already been initialized.")

    cred = credentials.Certificate(config_path)
    firebase_admin.initialize_app(cred)
    _firestore_client = firestore.client()

def get_collection(obj_type):
    """
    Get or create a collection for a specific object type.
    """
    collection_name = obj_type.lower()
    collection_ref = _firestore_client.collection(collection_name)
    
    if not collection_ref.get():
        collection_ref.create()
    
    logger.debug(f"Retrieved collection: {collection_name}")
    return collection_ref

def generate_combined_id(obj_type, obj_id):
    """
    Generate a combined ID in the format "{type}_{id}".
    """
    return f"{obj_type}_{obj_id}"

def retrieve_object(obj_type, obj_id):
    """
    Retrieve an object by its type and ID from the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    document = collection.document(combined_id).get()
    logger.debug(f"Retrieved object: {combined_id}")
    return document.to_dict() if document.exists else None

def check_nonexistent_ids(obj_type, id_list):
    """
    Checks and returns the list of IDs that do not exist in the database for a given object type.
    """
    collection = get_collection(obj_type)
    nonexistent_ids = []
    for obj_id in id_list:
        combined_id = generate_combined_id(obj_type, obj_id)
        document = collection.document(combined_id).get()
        if not document.exists:
            nonexistent_ids.append(obj_id)
            logger.debug(f"Nonexistent ID: {combined_id}")
    return nonexistent_ids

def bulk_store_documents(obj_type, key, documents):
    """
    Bulk store a list of documents in the collection corresponding to the object type.
    """
    collection = get_collection(obj_type)
    batch = _firestore_client.batch()

    for doc in documents:
        obj_id = doc.metadata.get(key)
        if obj_id is None:
            raise ValueError(f"Key '{key}' not found in metadata for one of the documents")

        combined_id = generate_combined_id(obj_type, obj_id)
        document_ref = collection.document(combined_id)

        document = {
            "type": obj_type,
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        batch.set(document_ref, document)
        logger.debug(f"Stored document: {combined_id}")

    batch.commit()

def store_object(obj_type, obj_id, page_content, metadata):
    """
    Store an object in the specified collection based on object type.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    document = {
        "type": obj_type,
        "page_content": page_content,
        "metadata": metadata
    }
    collection.document(combined_id).set(document)
    logger.debug(f"Stored object: {combined_id}")

def get_all_objects_of_type(obj_type):
    """
    Retrieve all objects of a specific type from the corresponding collection.
    """
    collection = get_collection(obj_type)
    documents = collection.stream()
    logger.debug(f"Retrieved all objects of type: {obj_type}")
    return [doc.to_dict() for doc in documents]

def check_object_exists(obj_type, obj_id):
    """
    Check if an object exists in the corresponding collection by its type and ID.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    document = collection.document(combined_id).get()
    logger.debug(f"Checked object existence: {combined_id}")
    return document.exists

def update_page_content(obj_type, obj_id, new_page_content):
    """
    Update the page_content of an object in the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    collection.document(combined_id).update({"page_content": new_page_content})
    logger.debug(f"Updated page content for object: {combined_id}")

def update_metadata(obj_type, obj_id, new_metadata):
    """
    Update the metadata of an object in the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    collection.document(combined_id).update({"metadata": new_metadata})
    logger.debug(f"Updated metadata for object: {combined_id}")

def delete_object(obj_type, obj_id):
    """
    Delete an object by its type and ID from the corresponding collection.
    """
    combined_id = generate_combined_id(obj_type, obj_id)
    collection = get_collection(obj_type)
    collection.document(combined_id).delete()
    logger.debug(f"Deleted object: {combined_id}")
