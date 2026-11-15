from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

model = ChatOllama(
    model='gemma3:1b'
)

prompt1 = PromptTemplate(
    template="Give me details about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a 5 line summary on the {text}",
    input_variables=['text']
)

parser = StrOutputParser()


chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({'topic':'Cricket'})

print(result)