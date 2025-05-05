from waitress import serve
from app import app

print("Server is running on http://localhost:8080")  # Log message

# Runs the app with a production-grade WSGI server
serve(app, host="0.0.0.0", port=8080)
