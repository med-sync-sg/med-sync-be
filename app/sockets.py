from flask_socketio import SocketIO

def register_socket_events(socketio: SocketIO):
    @socketio.on('connect')
    def handle_connect():
        print('Client connected')
        socketio.emit('message', {'msg': 'Welcome to the server!'})

    @socketio.on('disconnect')
    def handle_disconnect():
        print('Client disconnected')

    @socketio.on('transcription')
    def handle_transcription(data):
        print(f"Received audio data")
        # Simulate processing and send back a response
        audio_chunk = f"Transcribed text for: {data['audio']}"
        
        # Transcription of audio --> text
        
        socketio.emit('transcription_result', {'transcription': audio_chunk})