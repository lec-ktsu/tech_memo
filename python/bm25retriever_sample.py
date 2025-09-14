"""
BM25Retriever は langchain_community.retrievers.bm25 モジュールにある。
つまり、次のようにインポートする。

BM25Retriever は内部で rank_bm25 パッケージを使っている。
インストールしていなければ、以下のコマンドでインストールする。
>$ pip install rank_bm25
"""

# need to install rank_bm25: pip install rank_bm25
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter

# 1. サンプルドキュメントの作成
docs = [
    Document(page_content="Pythonは人気の高いプログラミング言語です。", meta_data="プログラミング言語", tag="プログラミング言語"),
    Document(page_content="東京は日本の首都であり、経済の中心地です。", meta_data="日本", tag=["東京", "経済", "首都"]),
    Document(page_content="寿司は日本の代表的な料理の一つです。", meta_data="日本", tag=["寿司", "料理"]),
    Document(page_content="富士山は日本一高い山です。", meta_data="日本", tag=["富士山", "山"]),
]

# 2. テキストを分割（BM25はBag-of-Wordsなので、適切に分割するのが良い）
text_splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)
print(type(split_docs), type(split_docs[0]))
print(split_docs[0])

# 3. BM25Retriever の初期化
retriever = BM25Retriever.from_documents(split_docs, k=3)

# 4. 検索クエリを使って関連ドキュメントを検索
query = "日本の山"
results = retriever.invoke(query, meta_data="日本")

# 5. 結果の表示
print(f"🔍 クエリ: {query}")
print("📄 検索結果:")
for i, doc in enumerate(results):
    print(f"{i+1}. {doc.page_content}")
