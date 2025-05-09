from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maximum 16MB upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Create upload directory if it doesn't exist

@app.route("/")
def index():
    return render_template("finetuning.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/api_settings")
def api_settings():
    return render_template("api_settings.html")  # Your HTML file content

@app.route('/save_api_settings', methods=['POST'])
def save_api_settings():
    # Get form data
    api_key = request.form.get('apiKey', '')
    temperature = float(request.form.get('temperature', 0.7))
    max_tokens = int(request.form.get('maxTokens', 1000))
    timeout = int(request.form.get('timeout', 30))
    urls = request.form.get('urls', '').splitlines()
    
    # Process files
    uploaded_files = []
    if 'files[]' in request.files:
        files = request.files.getlist('files[]')
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                uploaded_files.append(file_path)
    
    # Store settings in session or database (example uses a dictionary for now)
    settings = {
        'api_key': api_key,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'timeout': timeout,
        'urls': urls,
        'uploaded_files': uploaded_files
    }
    
    # Here you would typically save these settings to a database
    # For now, we'll just return them as confirmation
    return jsonify({
        'status': 'success',
        'message': 'Settings saved successfully',
        'settings': {k: v for k, v in settings.items() if k != 'api_key'}  # Don't return API key for security
    })

# Your existing message handling routes
@app.route('/send_message', methods=['POST'])
def handle_message():
    data = request.json
    user_message = data.get('message', '')
    processed_message = process_message(user_message)
    response = generate_response(processed_message)
    return jsonify({"response": response})

def process_message(message):
    return message

def generate_response(message):
    return f"You said: {message}. This is a simple echo response."

if __name__ == '__main__':
    app.run(debug=True)