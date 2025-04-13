import sqlite3

def get_user_info(username, password):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    # Vulnerable to SQL Injection
    query = "SELECT * FROM users WHERE username = '{}' AND password = '{}'".format(username, password)
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return result
