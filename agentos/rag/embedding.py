import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

__import__("pysqlite3")
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import sentence_transformers #sentence_transformers download model from huggingface
from typing import List
from agentos.rag.data import BaseData
from chromadb import Documents, Embeddings


class EmbeddingModel:
    def __init__(
        self,
        model_name:str,
        cache_dir:str=None,
        **kwargs
    ):
        self.model_name=model_name
        # "BAAI/bge-base-zh-v1.5"
        # cache_folder="/mnt/7T/xz"
        self.embedding_model= sentence_transformers.SentenceTransformer(  
            model_name, cache_folder=cache_dir,**kwargs
        )
    
    def __call__(self, input: Documents) -> Embeddings:
        return  [self.embedding_model.encode(text) for text in input]

    def embed_query(self, input: str) -> List[float]:
        """Required by ChromaDB for query embedding."""
        return self.embedding_model.encode(input).tolist()

    def embed_documents(self, input: Documents) -> Embeddings:
        """Required by ChromaDB for document embedding."""
        return [self.embedding_model.encode(text).tolist() for text in input]

    def encode(
        self,
        data:List[BaseData]
    ):
        content = [d.get_content() for d in data]
        return self.embedding_model.encode(content)
        


# import sentence_transformers 
# from typing import List
# from chromadb import Documents, Embeddings

# embedding_model= sentence_transformers.SentenceTransformer(  
#             "BAAI/bge-base-zh-v1.5", cache_folder="/mnt/7T/xz"
#         )
# # content=["我是中国人","我爱中国","我爱我的祖国"]
# content ="我是中国人"
# embeddings=embedding_model.encode(content)
# print(embeddings)
# print(embeddings.shape)
 