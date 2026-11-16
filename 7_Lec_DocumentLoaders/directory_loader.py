
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path="directory_loader_folder",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

for documents in docs:
    # print(documents.page_content)
    print(documents.metadata)



# print("Total documents:", len(docs))
# print(docs)

# print(docs[20].page_content)
# print(docs[20].metadata)





#   "*.pdf"	         All PDF files
#   "*.txt" 	     All TXT files
#   "*.docx"	     All DOCX files
#   "*"	             All files in the directory
#   "**/*.pdf"	     PDF files in all subdirectories
#   "**/*.txt"	     TXT files in all subdirectories
#   "*.csv"	         All CSV files