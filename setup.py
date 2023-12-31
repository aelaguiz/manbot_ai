from setuptools import setup, find_packages

setup(
    name='ai',  # Change to your library's name
    version='0.1.3',  # Update the version number for new releases
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
        'certifi==2023.11.17',
        'charset-normalizer==3.3.2',
        'dataclasses-json==0.6.3',
        'diskcache==5.6.3',
        'distro==1.9.0',
        'dnspython==2.4.2',
        'frozenlist==1.4.1',
        'h11==0.14.0',
        'httpcore==1.0.2',
        'httpx==0.26.0',
        'idna==3.6',
        'jsonpatch==1.33',
        'jsonpointer==2.4',
        'langchain==0.0.352',
        'langchain-community==0.0.6',
        'langchain-core==0.1.3',
        'langsmith==0.0.75',
        'loguru==0.7.2',
        'lxml==4.9.4',
        'marshmallow==3.20.1',
        'multidict==6.0.4',
        'mypy-extensions==1.0.0',
        'numpy==1.26.2',
        'openai==1.6.1',
        'packaging==23.2',
        'pinecone-client==2.2.4',
        'prompt-toolkit==3.0.43',
        'pydantic==2.5.3',
        'pydantic_core==2.14.6',
        'python-dateutil==2.8.2',
        'python-dotenv==1.0.0',
        'PyYAML==6.0.1',
        'requests==2.31.0',
        'six==1.16.0',
        'sniffio==1.3.0',
        'SQLAlchemy==2.0.24',
        'tenacity==8.2.3',
        'tqdm==4.66.1',
        'typing-inspect==0.9.0',
        'typing_extensions==4.9.0',
        'urllib3==2.1.0',
        'wcwidth==0.2.12',
        'yarl==1.9.4',
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
