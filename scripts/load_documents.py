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

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)

from lxml import etree as ET


def list_wordpress_documents(xml_file):
    # Parse the XML file using lxml
    parser = ET.XMLParser(recover=True)  # Set up the parser to recover from errors
    tree = ET.parse(xml_file, parser)

    root = tree.getroot()

    # Define the namespace
    namespace = {'wp': 'http://wordpress.org/export/1.2/'}

    channel = root.find('channel')
    if channel is not None:
        # Iterate over each child of 'channel'
        for child in channel:
            # Print the full XML of the child element
            print(ET.tostring(child, pretty_print=True).decode())
            print("\n\n")


    # # Iterate over each item/document
    # for item in root.findall('channel/item', namespace):
    #     title = item.find('title').text
    #     print("Document Title:", title)

def main():
    # 1. Load the wordpres
    # list_wordpress_documents('documents/innerconfidence.WordPress.2023-12-20.xml')

    #2. Load the books
    ##
    pass


if __name__ == "__main__":
    main()