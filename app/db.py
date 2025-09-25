import os
import psycopg2
from flask import current_app, g
from dotenv import load_dotenv

load_dotenv()

def get_db():
    if 'db' not in g:
        g.db = psycopg2.connect(
            host="db", # The docker-compose service name
            dbname=os.environ.get('POSTGRES_DB'),
            user=os.environ.get('POSTGRES_USER'),
            password=os.environ.get('POSTGRES_PASSWORD')
        )
    return g.db

def close_db(e=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()

def init_app(app):
    app.teardown_appcontext(close_db)