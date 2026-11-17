from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """Astronomy is the scientific study of celestial objects and phenomena in the universe. It helps us understand stars, planets, galaxies, moons, and other objects in space. The Sun is the closest star to Earth and provides the energy necessary for life. Our solar system consists of eight planets that revolve around the Sun. Earth is the only known planet that supports life because it has water, oxygen, and suitable temperatures. The Moon is Earth's natural satellite and plays an important role in ocean tides. 
 
Stars are massive balls of hot gases that produce light and heat through nuclear fusion. Galaxies are enormous collections of stars, gas, dust, and dark matter held together by gravity. The Milky Way is the galaxy in which our solar system is located. Astronomers use powerful telescopes to observe distant objects in space. Some telescopes are placed in space to capture clearer images without atmospheric interference. Black holes are regions of space where gravity is extremely strong, and even light cannot escape. Planets outside our solar system are known as exoplanets."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)


chunks = splitter.split_text(text)

print(chunks)