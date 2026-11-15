from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

model = ChatOllama(
    model='gemma3:1b'
)

prompt1 = PromptTemplate(
    template="Give 2 line joke {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Give me Explanation on {joke}",
    input_variables=['joke']
)

parser = StrOutputParser()


joke_gen_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke' : RunnablePassthrough(),
    'expln' : RunnableSequence(prompt2, model, parser)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'Astronomy'})

print(result)

final_chain.get_graph().print_ascii()