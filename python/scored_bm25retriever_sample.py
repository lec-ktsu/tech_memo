"""
ScoredBM25Retriever は LangChain の BaseRetriever を継承している。

LangChain v0.3 では、BaseRetriever は Pydantic v2 ベースになっているため、
フィールドを事前に宣言しておく必要がある。
具体的には、以下の 2 つが必要となる：

1. documents や k を __init__ ではなく Pydantic フィールドとして宣言する
2. カスタムの __init__ も極力使わず、__post_init__ 相当の処理を __init__ 後に行うように設計
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi
from typing import List
from pydantic import Field


class ScoredBM25Retriever(BaseRetriever):
    documents: List[Document] = Field()
    k: int = Field(default=4)

    # 明示的に許可
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
    }

    def __init__(self, **data):
        super().__init__(**data)

        # __init__ 後に必要な初期化
        self.corpus = [list(doc.page_content) for doc in self.documents]
        self.bm25 = BM25Okapi(self.corpus)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        tokenized_query = list(query)
        scores = self.bm25.get_scores(tokenized_query)

        # スコア付きでソート
        scored_docs = sorted(zip(self.documents, scores), key=lambda x: x[1], reverse=True)

        # 上位k件を返す（スコア付き）
        top_docs = []
        for doc, score in scored_docs[:self.k]:
            new_metadata = dict(doc.metadata)
            new_metadata["score"] = score
            top_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))

        return top_docs

    def invoke(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)



if __name__ == "__main__":
    from langchain_core.documents import Document

    docs = [
        Document(page_content="Pythonは人気の高いプログラミング言語です。", metadata={"id": 1}),
        Document(page_content="東京は日本の首都であり、経済の中心地です。", metadata={"id": 2}),
        Document(page_content="寿司は日本の代表的な料理の一つです。", metadata={"id": 3}),
        Document(page_content="富士山は日本一高い山です。", metadata={"id": 4}),
    ]

    retriever = ScoredBM25Retriever(documents=docs, k=4)
    query = "日本の山"
    results = retriever.invoke(query)

    print(f"🔍 クエリ: {query}")
    print("📄 検索結果（スコア付き）:")
    for i, doc in enumerate(results):
        print(f"{i+1}. score={doc.metadata['score']:.4f} | {doc.page_content}")

"""
* まとめ
| 課題                                  | 解決策                                                          |
| ----------------------------------- | ------------------------------------------------------------ |
| LangChain の `BaseRetriever` 継承時のエラー | フィールドを `pydantic.Field()` で明示的に定義                            |
| `__init__` を使う場合の注意                 | `super().__init__(**data)` を必ず呼び出す                           |
| 非標準型（BM25Okapiなど）                   | `model_config = {"arbitrary_types_allowed": True}` を設定する必要あり |

"""