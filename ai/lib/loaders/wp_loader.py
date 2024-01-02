import sys
import os
from typing import List
from bs4 import BeautifulSoup, Comment, NavigableString
import html
import json
from langchain.document_loaders import UnstructuredEPubLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
import logging
import html2text
from lxml import etree as ET



def sanitize_html(content):
    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')
    
    # Remove all comment tags
    for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
        comment.extract()

    for element in soup.contents:
        if isinstance(element, NavigableString) and element.strip():
            new_tag = soup.new_tag("p")
            new_tag.string = element
            element.replace_with(new_tag)

    
    # Get the text back
    clean_content = str(soup)
    
    return clean_content


class WPLoader(UnstructuredFileLoader):

    def load(self) -> List[Document]:
        h = html2text.HTML2Text()
        h.body_width = 0  # Set to 0 to prevent wrapping
        # h.single_line_break = True  # Ensures single line breaks are translated into markdown line breaks


        # Get base filename
        filename = os.path.basename(self.file_path)
        docs = []

        parser = ET.XMLParser(recover=True)  # Set up the parser to recover from errors
        tree = ET.parse(self.file_path, parser)

        root = tree.getroot()

        # Define the namespace
        namespace = {'wp': 'http://wordpress.org/export/1.2/'}

        # Iterate over each item/document
        for item in root.findall('channel/item', namespace):
            # print(ET.tostring(item, pretty_print=True).decode())
            title = item.find('title').text
            link = item.find('link').text
            guid = item.find('guid').text
            content = html.unescape(item.find('{http://purl.org/rss/1.0/modules/content/}encoded').text)
            clean_content = sanitize_html(content)
            markdown_content = h.handle(clean_content)

            print("Document Title:", title)
            print("Document Content:", clean_content)
            print("Document Content:", markdown_content)
            print("\n\n")

            doc = Document(page_content=markdown_content, metadata={
                "title": title if title else 'Unknown Title',
                "author": "Robbie Kramer",
                'type': 'wordpress',
                "filename": filename,
                "url": link,
                "guid": guid,
                "source": filename
            })

            docs.append(doc)
        
        return docs