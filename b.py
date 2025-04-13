def get_user_file(user_id):
    # Vulnerable to IDOR
    file_path = "/home/user_data/{}".format(user_id)
    with open(file_path, 'r') as file:
        return file.read()
