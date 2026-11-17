from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

mardown_text = """
# Astronomy

Astronomy is the scientific study of celestial objects and phenomena in the universe. It helps us understand stars, planets, galaxies, moons, and other objects in space.

## Solar System

Our solar system consists of the Sun and eight planets that revolve around it. The Sun is the closest star to Earth and provides energy necessary for life.

### Planets

1. Mercury is the smallest planet in our solar system.
2. Venus is the hottest planet.
3. Earth is the only known planet that supports life.
4. Mars is known as the Red Planet.
5. Jupiter is the largest planet.
6. Saturn is famous for its beautiful rings.
7. Uranus rotates on its side.
8. Neptune is the farthest planet from the Sun.

## Galaxies

A galaxy is a huge collection of stars, gas, dust, and dark matter held together by gravity.

### Milky Way

The Milky Way is the galaxy in which our solar system is located. It contains billions of stars and has a spiral structure."""




splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_text(mardown_text)

print(chunks[0])
