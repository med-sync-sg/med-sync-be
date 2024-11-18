from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO

socketio = SocketIO()  # Initialize SocketIO
def create_app():
    app = Flask(__name__)

    # Initialize SocketIO with the Flask app
    CORS(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # Register HTTP routes and WebSocket events
    with app.app_context():
        from .sockets import register_socket_events
        # app.register_blueprint(api, url_prefix='/api')
        register_socket_events(socketio)

    return app