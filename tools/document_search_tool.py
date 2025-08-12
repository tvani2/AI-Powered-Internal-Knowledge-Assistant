import os
import json
import glob
import shutil
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import Tool


@dataclass
class DocChunk:
    content: str
    metadata: Dict[str, Any]


class DocumentSearchTool:
    """Embeds and searches internal documents using OpenAI and ChromaDB."""

    def __init__(self, documents_dir: str = "documents", persist_dir: str = "vectorstore") -> None:
        try:
            load_dotenv()
        except Exception as e:
            print(f"Warning: Could not load .env file in document search tool: {e}")
            # Set environment variables manually (use placeholder)
            os.environ["OPENAI_API_KEY"] = "your-api-key-here"
            
        self.documents_dir = documents_dir
        self.persist_dir = persist_dir
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.index_path = os.path.join(self.persist_dir, "ingest_index.json")
        self.lock_path = os.path.join(self.persist_dir, ".ingest.lock")

    def _load_documents(self) -> List[str]:
        patterns = [
            os.path.join(self.documents_dir, "**", "*.txt"),
            os.path.join(self.documents_dir, "**", "*.md"),
        ]
        filepaths: List[str] = []
        for pattern in patterns:
            filepaths.extend(glob.glob(pattern, recursive=True))
        return sorted(set(filepaths))

    def _hash_file(self, path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def _load_index(self) -> Dict[str, str]:
        if os.path.isfile(self.index_path):
            try:
                with open(self.index_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_index(self, index: Dict[str, str]) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def _acquire_lock(self) -> None:
        os.makedirs(self.persist_dir, exist_ok=True)
        # Simple exclusive creation; raises if exists
        try:
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            raise RuntimeError("Another ingestion process appears to be running. Aborting.")

    def _release_lock(self) -> None:
        try:
            if os.path.isfile(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass

    def ingest(self, clear: bool = False) -> None:
        """Ingest documents into Chroma persistent store.

        - clear=True will wipe the existing persist_dir before re-ingesting
        - Incremental ingestion: only changed/new files are embedded
        """
        self._acquire_lock()
        try:
            if clear and os.path.isdir(self.persist_dir):
                shutil.rmtree(self.persist_dir)
                print(f"Cleared vectorstore at {self.persist_dir}")

            paths = self._load_documents()
            if not paths:
                print("No documents found to ingest.")
                return

            prev_index = self._load_index()
            new_index: Dict[str, str] = dict(prev_index)

            to_embed: List[DocChunk] = []
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)

            for path in paths:
                try:
                    file_hash = self._hash_file(path)
                    if prev_index.get(path) == file_hash:
                        continue  # unchanged
                    with open(path, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs = splitter.create_documents([text], metadatas=[{"source": path}])
                    for d in docs:
                        to_embed.append(DocChunk(content=d.page_content, metadata=d.metadata))
                    new_index[path] = file_hash
                except Exception as e:
                    print(f"Skipping {path}: {e}")

            if not to_embed:
                print("No changes detected. Vectorstore is up to date.")
                return

            texts = [c.content for c in to_embed]
            metadatas = [c.metadata for c in to_embed]

            if os.path.isdir(self.persist_dir) and os.path.isfile(os.path.join(self.persist_dir, "chroma.sqlite3")):
                store = Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)
                store.add_texts(texts=texts, metadatas=metadatas)
                store.persist()
            else:
                Chroma.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas,
                    persist_directory=self.persist_dir,
                )

            self._save_index(new_index)
            print(f"Ingested {len(texts)} new/updated chunks into {self.persist_dir}")
        finally:
            self._release_lock()

    def _get_store(self) -> Chroma:
        return Chroma(embedding_function=self.embeddings, persist_directory=self.persist_dir)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        store = self._get_store()
        docs_and_scores = store.similarity_search_with_score(query, k=k)
        return [
            {
                "content": d.page_content,
                "source": d.metadata.get("source"),
                "score": float(score),
            }
            for d, score in docs_and_scores
        ]

    def as_langchain_tool(self) -> Tool:
        def _run(query: str) -> str:
            results = self.search(query, k=5)
            return json.dumps(results, ensure_ascii=False)

        description = (
            "Semantic search over internal documents (HR policies, meeting notes, technical docs). "
            "Returns the most relevant passages with their sources and similarity scores as JSON."
        )
        return Tool(name="document_search", func=_run, description=description)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Document search tool")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into Chroma")
    parser.add_argument("--clear", action="store_true", help="Clear existing vectorstore before ingesting")
    parser.add_argument("--query", type=str, default=None, help="Run a sample query")
    parser.add_argument("--k", type=int, default=5, help="Number of results")
    args = parser.parse_args()

    tool = DocumentSearchTool()

    if args.ingest:
        tool.ingest(clear=args.clear)

    if args.query:
        print(json.dumps(tool.search(args.query, k=args.k), indent=2, ensure_ascii=False))
