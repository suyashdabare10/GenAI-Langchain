from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

loader = TextLoader("astronomy.txt", encoding="utf-8")

docs = loader.load()

text = docs[0].page_content

model = ChatOllama(
    model='gemma3:1b'
)

prompt= PromptTemplate(
    template="Generate a summary of the {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'text':text})

print(result)