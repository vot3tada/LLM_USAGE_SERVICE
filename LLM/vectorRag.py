from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, QueryBundle, \
    SummaryIndex, ServiceContext
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response.notebook_utils import display_response
from llama_index.core.response_synthesizers import ResponseMode, BaseSynthesizer
from llama_index.core.retrievers import RouterRetriever, QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import IndexNode
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
Settings.text_splitter = SentenceSplitter(chunk_size=1024)
# DB
client = qdrant_client.QdrantClient("http://localhost:6333")

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
transformations = [
    SentenceSplitter(chunk_size=256, chunk_overlap=64),
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

# RETRIVERS
vector_store = QdrantVectorStore(client=client, collection_name="documents", batch_size=256)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(documents,
                                        storage_context=storage_context)

vector_retriever = index.as_retriever(similarity_top_k=2)
docstore = SimpleDocumentStore()
docstore.add_documents(Settings.text_splitter.get_nodes_from_documents(documents))
bm25_retriever = BM25Retriever.from_defaults(
    docstore=docstore, similarity_top_k=2
)

vector_obj = IndexNode(
    index_id="vector", obj=vector_retriever, text="Vector Retriever"
)
bm25_obj = IndexNode(
    index_id="bm25", obj=bm25_retriever, text="BM25 Retriever"
)

summary_index = SummaryIndex(objects=[vector_obj, bm25_obj])
summary_retriever = summary_index.as_retriever()

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

# chat_mode=ChatMode.CONTEXT,
#     memory=memory,
#     response_mode=ResponseMode.REFINE,
#     system_prompt=(
#         "Ты - чатбот, которому подается контекст из базы знаний. Ты должен отвечать на вопросы, опираясь на базу знаний."
#     ),

query_engine = summary_index.as_chat_engine(
    chat_mode=ChatMode.CONTEXT,
    memory=memory,
    response_mode=ResponseMode.REFINE,
    system_prompt=(
        "Ты - чатбот, которому подается контекст из базы знаний. Ты должен отвечать на вопросы, опираясь на базу знаний."
    ),
)

while True:
    query = input("Question: ")

    vn = vector_retriever.retrieve(query)
    print("VECTOR")
    for node in vn:
        print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")
    print("BM")
    vb = bm25_retriever.retrieve(query)
    for node in vb:
        print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")
    print('SUM')
    vs = summary_retriever.retrieve(query)
    for node in vs:
        print(f"Score: {node.score:.2f} - {node.text}...\n-----\n")

    res = query_engine.chat(query)
    print(res)
