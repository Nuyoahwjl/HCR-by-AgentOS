import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)


from agentos.rag.data import BaseData
from typing import List
 
 
class CharacterSplit:
    def __init__(
        self,
        chunk_size:int,
        chunk_overlap:int=0
    ):
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap
    
    def split(
        self,
        data:BaseData
    )->List[BaseData]:
        
        meta_data=data.get_metadata()
        content = data.get_content()
        chunk_res = []

        i = 0
        while i < len(content):
            if i > 0:
                start = max(i - self.chunk_overlap, 0)
            else:
                start = i
            end = min(start + self.chunk_size, len(content))   

           
            chunk_res.append(BaseData(content=content[start:end],metadata=meta_data))
            
            i = end
            
        return chunk_res



class RowSplit:
    def __init__(
        self,
        chunk_row_size:int,
        chunk_overlap:int=0
    ):
        self.chunk_row_size=chunk_row_size
        self.chunk_overlap=chunk_overlap
        assert(chunk_overlap<chunk_row_size)

    def split(
        self,
        data:BaseData
    )->List[BaseData]:

        meta_data=data.get_metadata()
        content = data.get_content().split('\n')
        chunk_res = []

        begin = 0
        while begin < len(content):
            end = min(begin + self.chunk_row_size, len(content))
            chunk_res.append(BaseData(content='\n'.join(content[begin:end]),metadata=meta_data))
            begin=begin+self.chunk_row_size-self.chunk_overlap

        return chunk_res


class SemanticSplit:
    """Semantic chunking strategy for medical data.

    Groups related rows by medical condition category (cardiovascular, metabolic, etc.)
    rather than splitting row-by-row. Preserves clinical context.

    For CSV data: groups patient records by condition keywords.
    For PDF/text data: splits by section headings or paragraph boundaries.
    """

    # Medical condition categories and their keywords
    CONDITION_CATEGORIES = {
        "心血管": ["高血压", "冠心病", "心悸", "胸闷", "心电图", "血压", "心脏", "心脑血管"],
        "代谢": ["糖尿病", "高血脂", "血糖", "血脂", "胆固醇", "甘油三酯", "尿酸", "痛风", "BMI"],
        "消化": ["肝功能", "肾功能", "胃肠", "腹痛", "便秘", "腹泻", "肝", "肾"],
        "呼吸": ["咳嗽", "气短", "肺", "胸片", "呼吸"],
        "血液": ["血常规", "贫血", "血红蛋白", "白细胞", "血小板"],
        "肿瘤": ["肿瘤", "标志物", "癌症", "筛查", "AFP", "CEA"],
        "骨科": ["骨密度", "腰痛", "关节", "颈椎", "腰椎"],
        "眼科": ["视力", "眼底", "眼科"],
        "妇科": ["乳腺", "宫颈", "妇科"],
        "男科": ["前列腺", "生殖"],
    }

    def __init__(self, chunk_size: int = 3, chunk_overlap: int = 1):
        """Initialize SemanticSplit.

        Args:
            chunk_size: Maximum number of rows per chunk when grouping by category.
            chunk_overlap: Overlap between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _classify_row(self, row: str) -> str:
        """Classify a data row into a medical condition category.

        Returns the best-matching category name, or "其他" if no match.
        """
        scores = {}
        for category, keywords in self.CONDITION_CATEGORIES.items():
            score = sum(1 for kw in keywords if kw in row)
            if score > 0:
                scores[category] = score

        if scores:
            return max(scores, key=scores.get)
        return "其他"

    def split(self, data: BaseData) -> List[BaseData]:
        """Split data by semantic grouping.

        For CSV-like data: groups consecutive rows by condition category.
        For text/PDF data: splits by paragraph boundaries.

        Args:
            data: The input BaseData to split.

        Returns:
            List of semantically chunked BaseData objects.
        """
        meta_data = data.get_metadata()
        content = data.get_content()
        rows = content.split('\n')

        if len(rows) <= self.chunk_size:
            return [BaseData(content=content, metadata=meta_data)]

        # Classify each row
        categories = [self._classify_row(row) for row in rows]

        # Group consecutive rows with same category
        chunks = []
        current_group = [rows[0]]
        current_category = categories[0]

        for i in range(1, len(rows)):
            if categories[i] == current_category and len(current_group) < self.chunk_size:
                current_group.append(rows[i])
            else:
                # Save current group
                chunk_content = '\n'.join(current_group)
                chunk_meta = dict(meta_data)
                chunk_meta["category"] = current_category
                chunks.append(BaseData(content=chunk_content, metadata=chunk_meta))

                # Apply overlap
                if self.chunk_overlap > 0 and len(current_group) > self.chunk_overlap:
                    overlap_rows = current_group[-self.chunk_overlap:]
                else:
                    overlap_rows = []

                current_group = overlap_rows + [rows[i]]
                current_category = categories[i]

        # Don't forget the last group
        if current_group:
            chunk_content = '\n'.join(current_group)
            chunk_meta = dict(meta_data)
            chunk_meta["category"] = current_category
            chunks.append(BaseData(content=chunk_content, metadata=chunk_meta))

        return chunks

