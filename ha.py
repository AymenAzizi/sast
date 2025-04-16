import sqlite3
from flask import Flask, request

app = Flask(__name__)

# Simulated in-memory database
conn = sqlite3.connect(':memory:', check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT, password TEXT)")
cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'admin123')")
conn.commit()

@app.route('/login', methods=['GET'])
def login():
    username = request.args.get('username')
    password = request.args.get('password')

    Vulnerable to SQL Injection
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    print("Running query:", query)
    result = cursor.execute(query).fetchone()

    if result:
        return f"Welcome {result[1]}!"
    else:
        return "Login failed!"

if __name__ == '__main__':
    app.run(debug=True)
