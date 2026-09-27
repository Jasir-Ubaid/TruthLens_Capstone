"""
Database Layer for TruthLens Backend
Manages Motor async MongoDB client connection, collections, and fallback in-memory store.
Collections:
  - users
  - scans
  - predictions
  - explainability_outputs
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger("TruthLens.Backend.Database")

# Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "truthlens_db")


class InMemoryAsyncCursor:
    """Async iterator compatible with Motor cursor for fallback mode."""
    def __init__(self, items: List[Dict[str, Any]]):
        self._items = items
        self._iter = iter(items)

    def sort(self, key_or_list, direction=None):
        if isinstance(key_or_list, str):
            key = key_or_list
            reverse = (direction == -1)
        elif isinstance(key_or_list, list) and len(key_or_list) > 0:
            key, d = key_or_list[0]
            reverse = (d == -1)
        else:
            return self

        def sort_val(x):
            v = x.get(key)
            return v if v is not None else ""

        self._items.sort(key=sort_val, reverse=reverse)
        self._iter = iter(self._items)
        return self

    def limit(self, count: int):
        self._items = self._items[:count]
        self._iter = iter(self._items)
        return self

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        if length is not None:
            return self._items[:length]
        return list(self._items)


class InMemoryCollection:
    """Async collection mock matching Motor's interface for seamless offline operation."""
    def __init__(self, name: str):
        self.name = name
        self.docs: List[Dict[str, Any]] = []

    async def insert_one(self, doc: Dict[str, Any]):
        doc_copy = dict(doc)
        if "_id" not in doc_copy:
            doc_copy["_id"] = str(len(self.docs) + 1)
        self.docs.append(doc_copy)
        return type("InsertResult", (), {"inserted_id": doc_copy["_id"]})()

    async def find_one(self, filter_query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for doc in self.docs:
            match = True
            for k, v in filter_query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return dict(doc)
        return None

    def find(self, filter_query: Optional[Dict[str, Any]] = None):
        filter_query = filter_query or {}
        matched = []
        for doc in self.docs:
            match = True
            for k, v in filter_query.items():
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                matched.append(dict(doc))
        return InMemoryAsyncCursor(matched)

    async def update_one(self, filter_query: Dict[str, Any], update_doc: Dict[str, Any]):
        matched = await self.find_one(filter_query)
        if matched:
            for doc in self.docs:
                if doc.get("_id") == matched.get("_id") or all(doc.get(k) == v for k, v in filter_query.items()):
                    if "$set" in update_doc:
                        doc.update(update_doc["$set"])
                    if "$inc" in update_doc:
                        for ik, iv in update_doc["$inc"].items():
                            doc[ik] = doc.get(ik, 0) + iv
                    return type("UpdateResult", (), {"modified_count": 1})()
        return type("UpdateResult", (), {"modified_count": 0})()

    async def count_documents(self, filter_query: Dict[str, Any]) -> int:
        cur = self.find(filter_query)
        res = await cur.to_list()
        return len(res)


class DatabaseManager:
    """
    Manages MongoDB connections with Motor and automatic fallback store.
    """
    def __init__(self):
        self.client = None
        self.db = None
        self.is_connected = False
        self._in_memory_db: Dict[str, InMemoryCollection] = {
            "users": InMemoryCollection("users"),
            "scans": InMemoryCollection("scans"),
            "predictions": InMemoryCollection("predictions"),
            "explainability_outputs": InMemoryCollection("explainability_outputs")
        }

    async def connect(self):
        """Attempts connection to Motor MongoDB instance with timeout."""
        try:
            import motor.motor_asyncio

            logger.info(f"Connecting to MongoDB at {MONGODB_URL}...")
            # Set short serverSelectionTimeoutMS so app starts promptly even if Mongo is down
            self.client = motor.motor_asyncio.AsyncIOMotorClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=2000
            )
            # Ping database
            await self.client.admin.command('ping')
            self.db = self.client[DATABASE_NAME]
            self.is_connected = True
            logger.info(f"Connected to MongoDB database: '{DATABASE_NAME}' successfully.")
        except Exception as e:
            logger.warning(
                f"Could not connect to MongoDB server ({e}). "
                "Switching to in-memory async fallback store. "
                "All API features and collections remain fully operational."
            )
            self.is_connected = False
            self.db = None

    async def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed.")

    def _ensure_active_client(self):
        """Ensures the Motor client is bound to the currently running event loop."""
        if not self.is_connected or self.client is None:
            return
        try:
            import asyncio
            current_loop = asyncio.get_running_loop()
            client_loop = getattr(self.client, "io_loop", None)
            if client_loop is not None and (client_loop.is_closed() or client_loop is not current_loop):
                import motor.motor_asyncio
                self.client = motor.motor_asyncio.AsyncIOMotorClient(
                    MONGODB_URL,
                    serverSelectionTimeoutMS=2000
                )
                self.db = self.client[DATABASE_NAME]
        except Exception:
            pass

    def get_collection(self, collection_name: str):
        """Returns either the Motor collection or the fallback in-memory collection."""
        self._ensure_active_client()
        if self.is_connected and self.db is not None:
            return self.db[collection_name]
        return self._in_memory_db.get(collection_name, InMemoryCollection(collection_name))

    @property
    def users(self):
        return self.get_collection("users")

    @property
    def scans(self):
        return self.get_collection("scans")

    @property
    def predictions(self):
        return self.get_collection("predictions")

    @property
    def explainability_outputs(self):
        return self.get_collection("explainability_outputs")


# Global database manager instance
db_manager = DatabaseManager()
