import chromadb
import time
import uuid

# ChromaDB setup - default embedding use karega, no sentence_transformers needed!
chroma_client = chromadb.PersistentClient(path="./studymind_db")

collection = chroma_client.get_or_create_collection(
    name="student_memories"
)

def save_memory(user_id: str, content: str, memory_type: str = "general", importance: int = 5):
    memory_id = str(uuid.uuid4())
    collection.add(
        documents=[content],
        metadatas=[{
            "user_id": user_id,
            "type": memory_type,
            "timestamp": time.time(),
            "importance": importance
        }],
        ids=[memory_id]
    )
    return memory_id

def get_relevant_memories(user_id: str, query: str, limit: int = 6):
    try:
        results = collection.query(
            query_texts=[query],
            n_results=limit,
            where={"user_id": user_id}
        )
        if not results["documents"] or not results["documents"][0]:
            return []
        return results["documents"][0]
    except Exception:
        return []

def get_all_memories(user_id: str):
    try:
        results = collection.get(where={"user_id": user_id})
        memories = []
        for i, doc in enumerate(results["documents"]):
            memories.append({
                "content": doc,
                "type": results["metadatas"][i]["type"],
                "timestamp": results["metadatas"][i]["timestamp"],
                "importance": results["metadatas"][i]["importance"]
            })
        memories.sort(key=lambda x: x["timestamp"], reverse=True)
        return memories
    except Exception:
        return []

def forget_old_memories(user_id: str, days_threshold: int = 30):
    try:
        cutoff_time = time.time() - (days_threshold * 24 * 60 * 60)
        all_data = collection.get(where={"user_id": user_id})
        if not all_data["ids"]:
            return 0
        ids_to_delete = []
        for i, metadata in enumerate(all_data["metadatas"]):
            is_old = metadata["timestamp"] < cutoff_time
            is_low_importance = metadata["importance"] < 8
            if is_old and is_low_importance:
                ids_to_delete.append(all_data["ids"][i])
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
    except Exception:
        return 0