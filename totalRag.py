from typing import Dict
from llama_index.core import PromptTemplate, SimpleDirectoryReader, StorageContext, VectorStoreIndex, SummaryIndex, \
    Settings
from llama_index.core.chat_engine.types import ChatMode, AgentChatResponse
from llama_index.core.extractors import KeywordExtractor, TitleExtractor, QuestionsAnsweredExtractor, SummaryExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import TextNode, BaseNode
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.extractors.entity import EntityExtractor
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import qdrant_client

from configparser import ConfigParser

config = ConfigParser()
config.read('./config.ini', encoding='utf-8')

client = qdrant_client.QdrantClient(
    config['QDRANT']['URL'],
)

embed_model = HuggingFaceEmbedding(config['MODEL']['EMBEDDING'])

memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


async def _aextract_keywords_from_node(self, node: BaseNode) -> Dict[str, str]:
    """Extract keywords from a node and return it's metadata dict."""
    if self.is_text_node_only and not isinstance(node, TextNode):
        return {}

    context_str = node.get_content(metadata_mode=self.metadata_mode)
    keywords = await self.llm.apredict(
        PromptTemplate(
            template=f"""\
{{context_str}}. Перечисли {self.keywords} ключевых слов из отрывка документа через запятую. Ключевые слова: """
        ),
        context_str=context_str,
    )

    return {"excerpt_keywords": keywords.strip()}


KeywordExtractor._aextract_keywords_from_node = _aextract_keywords_from_node

model_path = config['MODEL']['LLM']
n_ctx = 3900
top_k = 30
top_p = 0.9
temperature = 0.1
repeat_penalty = 1.1

verbose = True if config['DEBUG']['ENABLE'] and config['DEBUG']['ENABLE'].lower() == 'true' else False

model = LlamaCPP(
    model_path=model_path,
    max_new_tokens=256,
    temperature=temperature,
    context_window=3900,
    model_kwargs={
        'n_ctx': n_ctx,
        'n_parts': 1,
        'n_gpu_layers': 35,
        'verbose': verbose,
        'main_gpu': 0,
        'top_k': top_k,
        'top_p': top_p
    })

Settings.llm = model
Settings.embed_model = embed_model

# CREATING
if config['DATA']['MODE'] not in ('CREATE', 'GET'):
    raise ValueError("MODE must be CREATE or GET")

if config['DATA']['MODE'] == 'CREATE':

    reader = SimpleDirectoryReader(input_dir='./data')

    documents = reader.load_data()

    if verbose:
        print(f"{len(documents)} documents found")

    transformations = [SentenceSplitter(chunk_size=256, chunk_overlap=32), ]

    if config['METADATA']['ENABLE'] and config['METADATA']['ENABLE'].lower() == 'true':
        device = 'cuda' if config['METADATA']['DEVICE'] and config['METADATA']['DEVICE'].lower() == 'cuda' else 'cpu'

        transformations += [
            TitleExtractor(nodes=3,
                           node_template="""Контекст: {context_str}. Напиши заголовок по данному отрывку. Заголовок: """,
                           combine_template="""{context_str}. На основании заголовков выше. Какой окончательный заголовок отрывка документа ? Заголовок: """
                           ),
            QuestionsAnsweredExtractor(questions=3),
            SummaryExtractor(summaries=["prev", "self"]),
            KeywordExtractor(keywords=10),
            EntityExtractor(prediction_threshold=0.5,
                            label_entities=False,
                            device=device),
        ]

    pipeline = IngestionPipeline(transformations=transformations)
    nodes = pipeline.run(
        documents=documents,
        in_place=True,
        show_progress=True
    )
    client.delete_collection("documents")
    vector_store = QdrantVectorStore(client=client, collection_name="documents", batch_size=256)

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, docstore=docstore)
    storage_context.persist(persist_dir="./bm_storage")

    index = VectorStoreIndex(nodes, storage_context=storage_context, store_nodes_override=True)
    vector_retriever = index.as_retriever(similarity_top_k=2)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore, similarity_top_k=2
    )
else:
    vector_store = QdrantVectorStore(client=client, collection_name="documents", batch_size=256)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
    vector_retriever = index.as_retriever(similarity_top_k=2)
    docstore = SimpleDocumentStore.from_persist_dir("bm_storage")
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

engine = summary_index.as_chat_engine(
    chat_mode=ChatMode.CONTEXT,
    memory=memory,
    response_mode=ResponseMode.REFINE,
    system_prompt=(
        "Ты - чатбот, которому подается контекст из базы знаний. "
        "Ты должен отвечать на вопрос опираясь на полученный контекст и предыдущие вопросы пользователя. Приоритет у последнего вопроса"
    ),
)


def send_quest(query: str) -> AgentChatResponse:
    if verbose:
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

    res = engine.chat(query)
    return res


def reset():
    engine.reset()
