import json
import sqlite3
from sqlite3 import Error

class DatabaseManager:
    def __init__(self, db_config_file):
        self.db_config_file = db_config_file
        self.databases = {}

    def load_config(self):
        with open(self.db_config_file, 'r') as f:
            config = json.load(f)
        for db in config['databases']:
            db_name = f"{db['name']}.db"
            self.databases[db_name] = db

    def build(self):
        for db_name, db_info in self.databases.items():
            conn = self.create_connection(db_name)
            if conn is not None:
                self.create_tables(conn, db_info['tables'])
                conn.close()

    def create_connection(self, db_name):
        try:
            conn = sqlite3.connect(db_name)
            return conn
        except Error as e:
            print(e)
        return None

    def create_tables(self, conn, tables):
        for table_name, table_info in tables.items():
            columns = table_info['columns']
            column_defs = ', '.join([f"{col_name} {col_type}" for col_name, col_type in columns.items()])
            sql = f"CREATE TABLE IF NOT EXISTS {table_name} ({column_defs});"
            self.create_table(conn, sql)

    def create_table(self, conn, create_table_sql):
        try:
            c = conn.cursor()
            c.execute(create_table_sql)
        except Error as e:
            print(e)

    # CRUD Operations
    def create(self, conn, table_name, data):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?'] * len(data))
        sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
        cur = conn.cursor()
        cur.execute(sql, list(data.values()))
        return cur.lastrowid

    def read(self, conn, table_name, conditions=None, columns='*'):
        sql = f"SELECT {columns} FROM {table_name}"
        if conditions:
            sql += " WHERE " + " AND ".join([f"{col} = ?" for col in conditions.keys()])
            cur = conn.cursor()
            cur.execute(sql, list(conditions.values()))
        else:
            cur = conn.cursor()
            cur.execute(sql)
        return cur.fetchall()

    # Additional CRUD functions (update, delete) follow a similar pattern
    # Implement as needed based on your application's requirements

    def update(self, conn, table_name, data, conditions):
        """
        Update records in the database.
        :param conn: Connection object
        :param table_name: String
        :param data: Dictionary of column values to update
        :param conditions: Dictionary of conditions for the update
        """
        set_clause = ', '.join([f"{column} = ?" for column in data.keys()])
        condition_clause = ' AND '.join([f"{column} = ?" for column in conditions.keys()])
        sql = f"UPDATE {table_name} SET {set_clause} WHERE {condition_clause}"
        cur = conn.cursor()
        cur.execute(sql, list(data.values()) + list(conditions.values()))
        conn.commit()
        return cur.rowcount

    def delete(self, conn, table_name, conditions):
        """
        Delete records from the database.
        :param conn: Connection object
        :param table_name: String
        :param conditions: Dictionary of conditions for the deletion
        """
        condition_clause = ' AND '.join([f"{column} = ?" for column in conditions.keys()])
        sql = f"DELETE FROM {table_name} WHERE {condition_clause}"
        cur = conn.cursor()
        cur.execute(sql, list(conditions.values()))
        conn.commit()
        return cur.rowcount

    def read_user_by_email(self, email):
        conn = self.create_connection('user_management.db')  # Use your actual DB name
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            return cur.fetchone()


    def create_user(self, user_data):
        conn = self.create_connection('user_management.db')  # Use your actual DB name
        with conn:
            cur = conn.cursor()
            columns = ', '.join(user_data.keys())
            placeholders = ', '.join(['?'] * len(user_data))
            cur.execute(f"INSERT INTO users ({columns}) VALUES ({placeholders})", tuple(user_data.values()))
            conn.commit()
            return cur.lastrowid


# Example usage
if __name__ == '__main__':
    db_manager = DatabaseManager('database.json')
    db_manager.load_config()
    db_manager.build()
