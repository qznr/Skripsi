import os
from flask import Flask

def create_app():
    app = Flask(__name__)

    # Initialize and register database functions
    import db
    db.init_app(app)

    # Register routes and commands
    import main
    import commands
    app.register_blueprint(main.bp)
    app.register_blueprint(commands.bp)

    return app