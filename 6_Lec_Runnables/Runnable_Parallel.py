from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

model = ChatOllama(
    model='gemma3:1b'
)

prompt1 = PromptTemplate(
    template="Give me 3 line tweet on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Give me 3 line Linkedin Post on {topic}",
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1, model, parser),
    'linkedin' : RunnableSequence(prompt2, model, parser)
})

result = parallel_chain.invoke({'topic':'Redshift'})

print(result)
