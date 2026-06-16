from enum import Enum


class ModelStyle(Enum):
    GORILLA = "gorilla"
    OPENAI_COMPLETIONS = "openai-completions"
    OPENAI_RESPONSES = "openai-responses"
    ANTHROPIC = "claude"
    MISTRAL = "mistral"
    GOOGLE = "google"
    AMAZON = "amazon"
    FIREWORK_AI = "firework_ai"
    NEXUS = "nexus"
    OSSMODEL = "ossmodel"
    COHERE = "cohere"
    WRITER = "writer"
    NOVITA_AI = "novita_ai"


class Language(Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"


class ReturnFormat(Enum):
    PYTHON = "python"
    JAVA = "java"
    JAVASCRIPT = "javascript"
    JSON = "json"
    VERBOSE_XML = "verbose_xml"
    CONCISE_XML = "concise_xml"
