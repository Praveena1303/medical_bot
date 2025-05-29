from passlib.context import CryptContext
import json

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def load_users():
    try:
        with open('users.json') as f:
            return json.load(f)
    except:
        return {}

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)

def create_user(username: str, password: str):
    users = load_users()
    users[username] = hash_password(password)
    with open('users.json', 'w') as f:
        json.dump(users, f)