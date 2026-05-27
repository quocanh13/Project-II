from flask import Flask, request
from flask_cors import CORS
from src.server.api import api
# python -m src.server.server

app = Flask(__name__)
CORS(app)

@app.before_request
def log_request():
    print(f"{request.method} {request.path}")
    
app.register_blueprint(api)

if __name__ == '__main__':
    app.run(debug=True)