from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model1 = ChatOllama(model="gemma3:1b")
model2 = ChatOllama(model="gemma3:1b")

prompt1 = PromptTemplate(
    template="Generate Notes about {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate a 5 Question Quiz on {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the Notes and Quiz together in one chat \n {notes} and {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes': prompt1 | model1 | parser,
    'quiz' : prompt2 | model2 | parser
})


merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

text = """
1. Introduction to Black Holes
A Black Hole is an extremely dense astronomical object in space whose gravitational force is so strong that nothing can escape from it, not even light. It is formed when a massive star reaches the end of its life cycle and collapses under its own gravitational force.

Black holes are among the most mysterious and fascinating objects in the universe. They have an extremely strong gravitational field because a huge amount of mass is concentrated into a very small region of space.

The term black hole is used because it does not emit or reflect light that can escape from its event horizon. Therefore, it appears black against the background of space.

Black holes are predicted by Albert Einstein's General Theory of Relativity, published in 1915. The theory explains how mass and energy curve space-time, producing gravity.
2. Formation of Black Holes
Black holes are commonly formed through the gravitational collapse of massive stars.

The formation process involves the following stages:
1. Birth of a Star: A star forms when a large cloud of gas and dust, mainly hydrogen, collapses under gravity.
2. Nuclear Fusion: The star produces energy through nuclear fusion in its core, converting hydrogen into helium.
3. Fuel Exhaustion: As the star ages, it eventually runs out of the nuclear fuel needed to maintain its internal pressure.
4. Gravitational Collapse: Gravity becomes stronger than the pressure supporting the star, causing its core to collapse.
5. Supernova Explosion: In many massive stars, the outer layers are expelled in a powerful supernova explosion.
6. Black Hole Formation: If the remaining core is sufficiently massive, it continues collapsing and forms a black hole.
Not every star becomes a black hole. Smaller stars generally become white dwarfs, while some massive stars leave behind neutron stars.
3. Structure of a Black Hole
A black hole consists of several important regions:
A. Singularity
The singularity is the central region of a classical black hole where matter is compressed to an extremely high density.

According to classical General Relativity, density and space-time curvature become infinite at the singularity. However, scientists believe that a complete theory of quantum gravity is needed to understand what actually happens there.
B. Event Horizon
The event horizon is the boundary surrounding a black hole beyond which nothing can escape, including light.

It is not a physical surface but a boundary in space-time. Once an object crosses this boundary, it cannot return to the outside universe.
C. Accretion Disk
An accretion disk is a rotating structure of gas, dust and other matter surrounding a black hole.

As matter moves toward the black hole, friction and gravitational effects heat it to extremely high temperatures. As a result, the disk can emit powerful X-rays and other forms of electromagnetic radiation.
D. Photon Sphere
The photon sphere is a region where gravity is strong enough to allow light to travel in circular orbits around the black hole.

These orbits are generally unstable, meaning a small disturbance can cause light to escape or fall inward.
E. Relativistic Jets
Some black holes produce powerful jets of high-energy particles that travel outward from regions near their poles.

These jets can extend thousands or even millions of light-years into space. They are associated with magnetic fields and matter around the black hole.
"""

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii()

