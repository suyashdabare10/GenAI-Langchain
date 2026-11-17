from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

code = """def calculate_sum(a, b):
    result = a + b
    return result

def calculate_subtraction(a, b):
    result = a - b
    return result

def calculate_multiplication(a, b):
    result = a * b
    return result"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(code)

print(chunks)

