from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

model = ChatOllama(
    model='gemma3:1b'
)

prompt= PromptTemplate(
    template="answer the following questions \n {question} from the following text - \n {text}",
    input_variables=['question','text']
)

parser = StrOutputParser()

url = 'https://www.flipkart.com/apple-macbook-air-m5-2026-m5-16-gb-512-gb-ssd-tahoe-mdhe4hn-a/p/itm8505e2f874525?pid=COMHZQX47WQ7YJZP&lid=LSTCOMHZQX47WQ7YJZPBZOHQJ&marketplace=FLIPKART&store=6bo%2Fb5g&srno=b_1_1&otracker=browse&fm=organic&iid=8ccc0cad-b02e-4058-8b8d-ebf9eba6f436.COMHZQX47WQ7YJZP.SEARCH&ppt=None&ppn=None&ssid=1f4euwdhc00000001794851055749&ov_redirect=true'

loader = WebBaseLoader(url) # if we want multiple webpage then we have to pass the list of urls. so then it will give one webpages as one document object

docs = loader.load()

chain = prompt | model | parser

# result = chain.invoke({'question':'what is the price of the macbook','text':docs})
result = chain.invoke({'question':'what is the specs of macbook','text':docs})

print(result)

# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
