from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model="gemma3:1b"
)

Prompt = PromptTemplate(
    template="Give me 5 line Detail about {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()


chain = Prompt | model | parser

result = chain.invoke({'topic':'Cricket'})

print(result)