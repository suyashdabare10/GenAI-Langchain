from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader('astronomy.pdf')

docs = loader.load()

# text = """Astronomy is the scientific study of celestial objects and phenomena in the universe. It helps us understand stars, planets, galaxies, moons, and other objects in space. The Sun is the closest star to Earth and provides the energy necessary for life. Our solar system consists of eight planets that revolve around the Sun. Earth is the only known planet that supports life because it has water, oxygen, and suitable temperatures. The Moon is Earth's natural satellite and plays an important role in ocean tides. Stars are massive balls of hot gases that produce light and heat through nuclear fusion. Galaxies are enormous collections of stars, gas, dust, and dark matter held together by gravity. The Milky Way is the galaxy in which our solar system is located. Astronomers use powerful telescopes to observe distant objects in space. Some telescopes are placed in space to capture clearer images without atmospheric interference. Black holes are regions of space where gravity is extremely strong, and even light cannot escape. Planets outside our solar system are known as exoplanets."""

splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,   
    separator=" "       #tells LangChain to split the text wherever it finds a space between words.
)

# chunks = splitter.split_text(splitter)
chunks = splitter.split_documents(docs)

print(chunks[0].page_content)
print(chunks[4].page_content)
# print(type(chunks))
# print(len(chunks))
# print(chunks[2])