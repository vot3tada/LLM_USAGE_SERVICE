from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.extractors import TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor, KeywordExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter, SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine

from typing import List

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import qdrant_client


documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
model_path = r'C:\Users\Sho\PycharmProjects\LLM_USAGE_SERVICE\saiga\model-mistral-q4_K.gguf'
n_ctx = 3900
top_k = 30
top_p = 0.9
temperature = 0.2
repeat_penalty = 1.1

model = LlamaCPP(
    model_path=model_path,
    max_new_tokens=256,
    temperature=temperature,
    context_window=3900,
    model_kwargs={
        'n_ctx': n_ctx,
        'n_parts': 1,
        'n_gpu_layers': 35,
        'verbose': True,
        'main_gpu': 0
    })

Settings.llm = model

Settings.embed_model = HuggingFaceEmbedding("cointegrated/rubert-tiny2")

reader = SimpleDirectoryReader(input_dir='./data/')

docs = reader.load_data()

transformations = [
    SentenceSplitter(chunk_size=256, chunk_overlap=128),
    TitleExtractor(nodes=5,
                   node_template="""\
Контекст: {context_str}. Сделай заголовок, который суммирует все \
уникальные объекты, названия или темы, найденные в контексте. Заголовок: """,
                   combine_template="""\
{context_str}. На основании вышеуказанных названий и содержания, \
какое полное название этого документа? Заголовок: """
                   ),
    # QuestionsAnsweredExtractor(questions=3),
    # SummaryExtractor(summaries=["prev", "self"]),
    # KeywordExtractor(keywords=10),
    # EntityExtractor(prediction_threshold=0.5,
    #                label_entities=False,
    #                device="cuda"),
]

pipeline = IngestionPipeline(transformations=transformations)
nodes = pipeline.run(
    documents=documents,
    in_place=True,
    show_progress=True,
)

client = qdrant_client.QdrantClient(
    "http://localhost:6333",
)

vector_store = QdrantVectorStore(client=client, collection_name="documents", batch_size=128)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

vector_retriever = VectorIndexRetriever(vector_index, similarity_top_k=3)
# custom_retriever = CustomRetriever(vector_retriever, bm25_retriever, mode="OR")

# query_engine = RetrieverQueryEngine(
#    retriever=vector_retriever,
#    response_synthesizer=get_response_synthesizer(),
# )
query_engine = vector_index.as_query_engine()

while True:
    query = input("Question: ")
    response = query_engine.query(query)
    print("------------------------------------")
    print(response)
    documents = vector_retriever.retrieve(query)
    print(f'Количество релевантных нод: {len(documents)}\n\n')

    for doc in documents:
        print(f'Оценка релевантности: {doc.score}\n')
        print(f'Содержание ноды: {doc.node.get_content()}\n\n')
    print("------------------------------------")
