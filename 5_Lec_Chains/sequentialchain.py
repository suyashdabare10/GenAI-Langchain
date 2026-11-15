from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(model="gemma3:1b")

prompt1 = PromptTemplate(
    template="Give me detail about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Generate a 5-line summary on {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"topic": "Black Hole"})

print(result)


chain.get_graph().print_ascii()