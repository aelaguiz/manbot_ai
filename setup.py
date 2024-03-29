from setuptools import setup, find_packages

from ai.version import __version__

setup(
    name='ai',  # Change to your library's name
    version=__version__,
    author='Amir Elaguizy',  # Replace with your name
    author_email='aelaguiz@gmail.com',  # Replace with your email
    description='AI utility library',  # Provide a short description
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/aelaguiz', 
    packages=find_packages(),
    install_requires=[
        'aiohttp==3.9.1',
        'aiosignal==1.3.1',
        'annotated-types==0.6.0',
        'anyio==4.2.0',
        'attrs==23.1.0',
        'backoff==2.2.1',
        'beautifulsoup4==4.12.2',
        'bs4==0.0.1',
        'CacheControl==0.13.1',
        'cachetools==5.3.2',
        'certifi==2023.11.17',
        'cffi==1.16.0',
        'chardet==5.2.0',
        'charset-normalizer==3.3.2',
        'click==8.1.7',
        'cryptography==41.0.7',
        'dataclasses-json==0.6.3',
        'diskcache==5.6.3',
        'distro==1.9.0',
        'dnspython==2.4.2',
        'EbookLib==0.18',
        'emoji==2.9.0',
        'filetype==1.2.0',
        'firebase==4.0.1',
        'firebase-admin==6.3.0',
        'frozenlist==1.4.1',
        'google-api-core==2.15.0',
        'google-api-python-client==2.111.0',
        'google-auth==2.25.2',
        'google-auth-httplib2==0.2.0',
        'google-cloud-core==2.4.1',
        'google-cloud-firestore==2.14.0',
        'google-cloud-storage==2.14.0',
        'google-crc32c==1.5.0',
        'google-resumable-media==2.7.0',
        'googleapis-common-protos==1.62.0',
        'grpcio==1.60.0',
        'grpcio-status==1.60.0',
        'h11==0.14.0',
        'html2text==2020.1.16',
        'httpcore==1.0.2',
        'httplib2==0.22.0',
        'httpx==0.26.0',
        'idna==3.6',
        'joblib==1.3.2',
        'jsonpatch==1.33',
        'jsonpath-python==1.0.6',
        'jsonpointer==2.4',
        'langchain==0.1.5',
        'langchain-community==0.0.17',
        'langchain-core==0.1.18',
        'langchain-openai==0.0.5',
        'langdetect==1.0.9',
        'langsmith==0.0.86',
        'loguru==0.7.2',
        'lxml==4.9.4',
        'markdown-it-py==3.0.0',
        'marshmallow==3.20.1',
        'mdurl==0.1.2',
        'msgpack==1.0.7',
        'multidict==6.0.4',
        'mypy-extensions==1.0.0',
        'nltk==3.8.1',
        'numpy==1.26.2',
        'openai==1.11.0',
        'packaging==23.2',
        'pgvector==0.2.4',
        'pinecone-client==2.2.4',
        'prompt-toolkit==3.0.43',
        'proto-plus==1.23.0',
        'protobuf==4.25.1',
        'psycopg2==2.9.9',
        'pyasn1==0.5.1',
        'pyasn1-modules==0.3.0',
        'pycparser==2.21',
        'pydantic==2.5.3',
        'pydantic_core==2.14.6',
        'Pygments==2.17.2',
        'PyJWT==2.8.0',
        'pyparsing==3.1.1',
        'python-dateutil==2.8.2',
        'python-dotenv==1.0.0',
        'python-iso639==2023.12.11',
        'python-magic==0.4.27',
        'PyYAML==6.0.1',
        'rapidfuzz==3.6.1',
        'regex==2023.12.25',
        'requests==2.31.0',
        'rich==13.7.0',
        'rsa==4.9',
        'six==1.16.0',
        'sniffio==1.3.0',
        'soupsieve==2.5',
        'SQLAlchemy==2.0.24',
        'tabulate==0.9.0',
        'tenacity==8.2.3',
        'tiktoken==0.5.2',
        'tqdm==4.66.1',
        'typing-inspect==0.9.0',
        'typing_extensions==4.9.0',
        'unstructured==0.11.6',
        'unstructured-client==0.15.1',
        'uritemplate==4.1.1',
        'urllib3==2.1.0',
        'wcwidth==0.2.12',
        'wrapt==1.16.0',
        'yarl==1.9.4'
    ],
    classifiers=[
        # Choose your license as you wish (should match "license" above)
         'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    python_requires='>=3.6',
    # Add any additional URLs that are relevant to your project as a dictionary under 'project_urls'
    project_urls={  
        'Source': 'https://github.com/aelaguiz',  
    },
)
