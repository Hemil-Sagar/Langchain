from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = "books",
    glob= '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()

# print(docs[17].page_content)
# print(docs[17].metadata)

for document in docs:
    print(document.metadata)