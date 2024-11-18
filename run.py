from flask_socketio import SocketIO
import app

flask_app = app.create_app()
socketio = SocketIO(app)

if __name__ == '__main__':
    socketio.run(flask_app, host='0.0.0.0', port=5000, debug=True)