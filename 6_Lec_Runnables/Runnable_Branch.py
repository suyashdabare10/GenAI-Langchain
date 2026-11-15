from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch

model = ChatOllama(
    model='gemma3:1b'
)

prompt1 = PromptTemplate(
    template="Give me detail 500 lines about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="summarize the {text} to 5 lines",
    input_variables=['text']
)

parser = StrOutputParser()


generate_detail_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>500 , RunnableSequence(prompt2, model, parser )),
    (RunnablePassthrough())
)

final_chain = RunnableSequence(generate_detail_chain, branch_chain)

result = final_chain.invoke({'topic':'Bigbang thoery'})

# print(result)

final_chain.get_graph().print_ascii()