from flask import Flask

def create_app():
    print("Creating Flask app...")  # Step 1: App creation
    app = Flask(__name__)

    # Import and register routes
    from .routes import main as main_blueprint
    print("Registering blueprint...")  # Step 2: Registering routes
    app.register_blueprint(main_blueprint)

    return app

