import logging

def format_docs(docs):
    logger = logging.getLogger(__name__)
    logger.debug(f"Formatting docs: {docs}")
    res = "\n\n".join([_format_doc(d) for d in docs])

    logger = logging.getLogger(__name__)
    # logger.debug(f"Formatted docs: {res}")

    return res

def _format_doc(doc):
    if doc.metadata['type'] == 'wordpress':
        return _format_wordpress(doc)
    elif doc.metadata['type'] == 'discord':
        return _format_discord(doc)
    elif doc.metadata['type'] == 'whatsapp_chat':
        return _format_whatsapp(doc)
    elif doc.metadata['type'] == 'book':
        return _format_book(doc)

    logger = logging.getLogger(__name__)
    logger.error(f"Unknown doc type: {doc.metadata['type']}")

def _format_book(doc):
    return f"""### Book
Title: {doc.metadata['title']}
Author: {doc.metadata['author']}

Summary: \"\"\"
{doc.page_content}
\"\"\""""

def _format_wordpress(doc):
    return f"""### Wordpress article
Title: {doc.metadata['title']}
Author: {doc.metadata['author']}
URL: {doc.metadata['url']}

Text: \"\"\"
{doc.page_content}
\"\"\""""

def _format_discord(doc):
    return f"""### Discord message
Topic: {doc.metadata['title']}
Filename: {doc.metadata['filename']}
Participant: {doc.metadata['participants']}
Timestamp: {doc.metadata['timestamp']}

Chat: \"\"\"
{doc.page_content}
\"\"\""""

def _format_whatsapp(doc):
    return f"""### Whatsapp conversation
Chat: \"\"\"
{doc.page_content}
\"\"\""""