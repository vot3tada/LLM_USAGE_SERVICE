from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.retrievers import RouterRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.tools import RetrieverTool
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import qdrant_client

### SETUP
# DATA
documents = SimpleDirectoryReader(input_dir="data").load_data()

# LLM
model_path = r'C:\Users\DenKach\Desktop\LLM_USAGE_SERVICE\LLM\model-q4_K.gguf'
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
    }
)

Settings.llm = model

# EMBEDING
embed_model = HuggingFaceEmbedding("cointegrated/rubert-tiny2")
Settings.embed_model = embed_model

# DB
client = qdrant_client.QdrantClient("http://localhost:6333")
vector_store = QdrantVectorStore(
    collection_name="mark_collection",
    client=client
)

### PROCESSING
# NODES
# transformations = [
#    TitleExtractor(nodes=5),
#    QuestionsAnsweredExtractor(questions=3),
#    SummaryExtractor(summaries=["prev", "self"]),
#    KeywordExtractor(keywords=10),
#    EntityExtractor(prediction_threshold=0.5,
#                    label_entities=False,
#                    device="cpu"), ]
#
# pipeline = IngestionPipeline(transformations=transformations)
# nodes = pipeline.run(documents=documents)
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(documents)

# RETRIVERS
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes, storage_context=storage_context
)
vector_retriever = index.as_retriever()
docstore = SimpleDocumentStore()
docstore.add_documents(nodes)
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, similarity_top_k=3
)

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=3,
    num_queries=1,  # set this to 1 to disable query generation
    mode=FUSION_MODES.RECIPROCAL_RANK,
    use_async=True,
    verbose=True,
    # query_gen_prompt="...",  # we could override the query generation prompt here
)

nodes_with_scores = vector_retriever.retrieve(
    "Что такое торговая марка ?"
)

for node in nodes_with_scores:
    print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")


query_engine = index.as_query_engine(llm=model)

response = query_engine.query(
    "Что такое торговая марка ?"
)
print(response)
