from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.indices.vector_store import VectorIndexRetriever
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

documents = SimpleDirectoryReader("data").load_data()

# bge embedding model
model_path = r'C:\Users\Sho\PycharmProjects\LLM_USAGE_SERVICE\saiga\model-q4_K.gguf'
n_ctx = 2000
top_k = 30
top_p = 0.9
temperature = 0.2
repeat_penalty = 1.1

model = LlamaCPP(
    model_path=model_path,
    model_kwargs={
        'n_ctx': n_ctx,
        'n_parts': 1,
        'n_gpu_layers': 35,
        'verbose': True,
        'main_gpu': 0
    })

Settings.llm = model

Settings.embed_model = HuggingFaceEmbedding("cointegrated/rubert-tiny2")

index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = index.as_query_engine()
query = "Кто автор романа?"
response = query_engine.query(query)
print(response)

# Создаем ретривер
retriever = VectorIndexRetriever(index) # хотя проще создать index.as_retriever()

# Задаем запрос и ищем релевантные ноды
documents = retriever.retrieve(query)

print(f'Количество релевантных нод: {len(documents)}\n\n')

for doc in documents:
    print(f'Оценка релевантности: {doc.score}\n')
    print(f'Содержание ноды: {doc.node.get_content()[:200]}\n\n')


