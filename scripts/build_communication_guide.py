import sys
import json
import os

import html
import html2text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

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

from pydantic import BaseModel, Field
from typing import Dict, List

class Introduction(BaseModel):
    overview_of_the_author: List[str] = Field(..., alias="Overview of the Author")

class GeneralWritingStyle(BaseModel):
    voice_and_tone: List[str] = Field(..., alias="Voice and Tone")
    sentence_structure: List[str] = Field(..., alias="Sentence Structure")
    paragraph_structure: List[str] = Field(..., alias="Paragraph Structure")

class VocabularyAndLanguageUse(BaseModel):
    word_choice: List[str] = Field(..., alias="Word Choice")
    language_style: List[str] = Field(..., alias="Language Style")
    metaphors_and_similes: List[str] = Field(..., alias="Metaphors and Similes")

class CharacterizationAndDialogue(BaseModel):
    character_development: List[str] = Field(..., alias="Character Development")
    dialogue_style: List[str] = Field(..., alias="Dialogue Style")

class NarrativeElements(BaseModel):
    pacing_and_rhythm: List[str] = Field(..., alias="Pacing and Rhythm")
    story_structure: List[str] = Field(..., alias="Story Structure")
    themes_and_motifs: List[str] = Field(..., alias="Themes and Motifs")

class EmotionalContext(BaseModel):
    evoking_emotions: List[str] = Field(..., alias="Evoking Emotions")
    atmosphere_and_setting: List[str] = Field(..., alias="Atmosphere and Setting")

class PersonalInsightsAndQuirks(BaseModel):
    authors_influences: List[str] = Field(..., alias="Author's Influences")
    personal_quirks: List[str] = Field(..., alias="Personal Quirks")

class ExamplesAndAnalysis(BaseModel):
    excerpts_from_works: List[str] = Field(..., alias="Excerpts from Works")
    detailed_analysis: List[str] = Field(..., alias="Detailed Analysis")

class WritersStyleGuide(BaseModel):
    introduction: Introduction = Field(..., alias="1. Introduction")
    general_writing_style: GeneralWritingStyle = Field(..., alias="2. General Writing Style")
    vocabulary_and_language_use: VocabularyAndLanguageUse = Field(..., alias="3. Vocabulary and Language Use")
    characterization_and_dialogue: CharacterizationAndDialogue = Field(..., alias="4. Characterization and Dialogue")
    narrative_elements: NarrativeElements = Field(..., alias="5. Narrative Elements")
    emotional_context: EmotionalContext = Field(..., alias="6. Emotional Context")
    personal_insights_and_quirks: PersonalInsightsAndQuirks = Field(..., alias="7. Personal Insights and Quirks")
    examples_and_analysis: ExamplesAndAnalysis = Field(..., alias="8. Examples and Analysis")

    class Config:
        allow_population_by_field_name = True


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
        | PydanticOutputParser(pydantic_object=WritersStyleGuide)
    )

    guidelog = open('guidelog.txt', 'w')
    comguides = []
    for doc in docs:
        new_comguide = chain.invoke({
            "input": doc.page_content,
            "comguide": ai_defaults.comguide_default,
            "example": ai_defaults.comguide_example,
        }, config={'callbacks': [lmd]})


        guidelog.write(f"\n\n********************************************************\n")
        guidelog.write(f"Document: {doc.metadata['title']}\n")
        guidelog.write(f"Content: {doc.page_content}\n\n")
        guidelog.write(f"Comguide: {new_comguide}\n\n")

        guidelog.flush()
        comguides.append(new_comguide.dict())

        with open(comguide_path, "w") as f:
            json.dump(comguides, f)

        comguide = new_comguide


if __name__ == "__main__":
    main()