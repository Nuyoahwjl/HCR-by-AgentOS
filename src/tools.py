import sys
import os

# 获取当前文件所在目录和项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 导入所需模块
from agentos.rag.embedding import EmbeddingModel
from agentos.rag.store import ChromaDB
from agentos.rag.bm25_retriever import BM25Retriever
from agentos.rag.hybrid_retriever import HybridRetriever
from agentos.rag.rerank import Rerank
from agentos.rag.data import BaseData, merge_content
from config.settings import Config
from src.query_decomposer import QueryDecomposer
from src.ontology import MedicalOntology

# 初始化嵌入模型
embedding = EmbeddingModel(
    model_name="BAAI/bge-base-zh-v1.5",
    # cache_dir="/mnt/7T/xz"
)

# 加载两个向量数据库
v1 = ChromaDB.load_document(
    embedding_model=embedding,
    dir=project_root + Config.VECTORSTORE1_PATH
)

v2 = ChromaDB.load_document(
    embedding_model=embedding,
    dir=project_root + Config.VECTORSTORE2_PATH
)

# 构建 BM25 索引 (从 ChromaDB 中获取全部文档)
_corpus_v1 = v1.get_all_documents()
bm25_v1 = BM25Retriever(corpus=_corpus_v1)

_corpus_v2 = v2.get_all_documents()
bm25_v2 = BM25Retriever(corpus=_corpus_v2)

# 初始化混合检索器
hybrid_v1 = HybridRetriever(
    chroma_db=v1,
    bm25_retriever=bm25_v1,
    fusion_method=Config.HYBRID_FUSION,
    alpha=Config.HYBRID_ALPHA,
    rrf_k=Config.RRF_K
)

hybrid_v2 = HybridRetriever(
    chroma_db=v2,
    bm25_retriever=bm25_v2,
    fusion_method=Config.HYBRID_FUSION,
    alpha=Config.HYBRID_ALPHA,
    rrf_k=Config.RRF_K
)

# 初始化 Reranker (延迟加载，首次使用时初始化)
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Rerank(model_name=Config.RERANKER_MODEL)
    return _reranker

# 初始化医学同义词和查询分解器
ontology = MedicalOntology()
decomposer = QueryDecomposer(ontology=ontology)


class search_by_id:
    """
    根据患者ID查询其曾经的体检信息
    """
    def __init__(self):
        pass

    def run(self, ID: str):
        """
        search_by_id:根据患者id在数据库查询其曾经的体检信息
        Args:
            ID (str): 唯一标识用户的六位数ID
        Returns:
            str: 查询结果或提示信息
        """
        result = v1.query_data("患者ID:{}".format(ID), query_num=1)
        result = merge_content(result)
        check = result[5:11]
        if check != ID:
            result = "没有找到该患者曾经的的体检信息"
        return result


class search_by_other:
    """
    根据患者个人信息查询相似病人体检信息 (使用混合检索+重排序)
    """
    def __init__(self):
        pass

    def run(self, num: int, user_info: str):
        """
        search_by_other:根据患者个人信息在数据库查询相似病人体检信息
        使用混合检索(BM25+Dense)和reranker提升检索质量。
        Args:
            num (int): 需要查询的相似体检信息数量(不超过5)
            user_info (str): 用户输入的除ID外全部个人信息，格式为"性别,年龄(岁),身高(cm),体重(kg),既往病史,症状"
        Returns:
            str: 查询结果
        """
        query_text = user_info
        try:
            reranker = get_reranker()
            results = hybrid_v1.retrieve(
                query=query_text,
                top_k=int(num),
                reranker=reranker,
                rerank_top_k=int(num)
            )
        except Exception:
            # Fallback to pure vector if hybrid fails
            results = v1.query_data(query_text, query_num=int(num))

        if isinstance(results[0], BaseData) if results else False:
            return merge_content(results)
        return merge_content(results)


class search_by_hybrid:
    """
    使用查询分解+混合检索查询相似病人体检信息 (高级检索)
    """
    def __init__(self):
        pass

    def run(self, num: int, gender: str, age: str, height: str, weight: str, medical_history: str, symptoms: str):
        """
        search_by_hybrid:使用查询分解和混合检索查询相似病人体检信息
        Args:
            num (int): 需要查询的相似体检信息数量(不超过5)
            gender (str): 性别
            age (str): 年龄
            height (str): 身高(cm)
            weight (str): 体重(kg)
            medical_history (str): 既往病史
            symptoms (str): 症状
        Returns:
            str: 查询结果
        """
        user_info = {
            "gender": gender,
            "age": age,
            "height": height,
            "weight": weight,
            "medical_history": medical_history,
            "symptoms": symptoms
        }

        # 查询分解
        sub_queries = decomposer.decompose(user_info)

        # 对每个子查询执行混合检索
        all_results = {}
        seen_contents = set()

        for sq in sub_queries:
            facet = sq["facet"]
            query = sq["query"]
            try:
                results = hybrid_v1.retrieve(
                    query=query,
                    top_k=3,
                    reranker=get_reranker(),
                    rerank_top_k=2
                )
            except Exception:
                results = v1.query_data(query, query_num=3)

            if facet not in all_results:
                all_results[facet] = []

            for r in results:
                content = r.get_content() if isinstance(r, BaseData) else r
                if content not in seen_contents:
                    seen_contents.add(content)
                    all_results[facet].append(content)

        # 合并去重结果
        merged = decomposer.merge_sub_query_results(all_results, max_total=int(num))
        return "\n".join(merged) if merged else "未找到相似体检信息"


class recommend_by_age:
    """
    根据患者年龄阶段推荐不同的体检项目
    """
    def __init__(self):
        pass

    def run(self, age: int):
        """
        recommend_by_age:根据患者的年龄阶段推荐不同的体检项目
        Args:
            age (int): 患者年龄
        Returns:
            str: 推荐的体检项目
        """
        age = int(age)
        if age <= 18:
            result = "建议关注生长发育、视力、营养、心理健康等方面的检查"
        elif age <= 40:
            result = "建议关注体重、血压、血糖、血脂、甲状腺功能、颈椎腰椎的检查"
        elif age <= 60:
            result = "建议关注心脑血管、肿瘤标志物、骨密度及其他慢性病的检查"
        else:
            result = "建议加强肿瘤筛查、心脑血管检查、胃肠镜检查，同时进行认知功能衰退评估"
        return result


class recommend_by_gender:
    """
    根据患者性别推荐不同的体检项目
    """
    def __init__(self):
        pass

    def run(self, gender: str):
        """
        recommend_by_gender:根据患者的性别推荐不同的体检项目
        Args:
            gender (str): 患者性别(male/female)
        Returns:
            str: 推荐的体检项目
        """
        if gender == "male":
            result = "建议男性关注前列腺、肝脏、心脑血管、肺部、生殖系统等方面的检查"
        elif gender == "female":
            result = "建议女性关注乳腺、宫颈、妇科等方面的检查"
        else:
            result = "性别输入有误，请输入'male'或'female'"
        return result
