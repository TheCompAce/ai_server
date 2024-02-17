import json
import sqlite3
import os
import pickle
import base64
import hashlib

class Cache:
    def __init__(self, db_path='cache.db'):
        self.cache = {}
        self.db_path = db_path
        self.init_db()
        self.load_from_database()

    def get(self, key):
        hashed_key = self._hash_key(key)
        item = self.cache.get(hashed_key)
        if item and item.get("is_binary"):
            return pickle.loads(base64.b64decode(item.get("value")))
        return item.get("value") if item else None

    def set(self, key, value, is_value_binary=False):
        hashed_key = self._hash_key(key)
        self.cache[hashed_key] = {"value": value, "is_binary": is_value_binary}
        self.save_to_database()

    def _hash_key(self, key):
        # If the key is already binary, use it directly; otherwise, serialize it first
        key_bytes = key if isinstance(key, bytes) else pickle.dumps(key)
        return hashlib.sha256(key_bytes).hexdigest()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache(
                key TEXT PRIMARY KEY,
                value TEXT,
                is_binary INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def save_to_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for key, item in self.cache.items():
            value = item["value"]
            is_binary = item["is_binary"]
            serialized_value = base64.b64encode(pickle.dumps(value)).decode("utf-8") if is_binary else value
            cursor.execute('''
                INSERT OR REPLACE INTO cache(key, value, is_binary) VALUES(?, ?, ?)
            ''', (key, serialized_value, int(is_binary)))
        conn.commit()
        conn.close()

    def load_from_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT key, value, is_binary FROM cache')
        rows = cursor.fetchall()
        for key, value, is_binary in rows:
            deserialized_value = pickle.loads(base64.b64decode(value)) if is_binary else value
            self.cache[key] = {"value": deserialized_value, "is_binary": is_binary}
        conn.close()
