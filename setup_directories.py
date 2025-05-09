"""
Setup script to create necessary folders and files for the Flask application.
Run this file once to create the required directory structure.
"""

import os
import shutil

def create_directory(path):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

def create_file(path, content=""):
    """Create a file with optional content"""
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")

def main():
    """Set up directory structure for the application"""
    # Create main directories
    create_directory('static')
    create_directory('templates')
    create_directory('tmp')
    create_directory('articles')
    create_directory('vector_index')
    
    # Create an empty .env file if it doesn't exist
    create_file('.env', 'OPENAI_API_KEY=\nFLASK_SECRET_KEY=your_secret_key\n')
    
    # Create a .gitignore file
    create_file('.gitignore', """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env

# Virtual environment
venv/
ENV/

# Flask
instance/
.webassets-cache

# Application specific
tmp/
articles/
vector_index/

# IDE
.idea/
.vscode/
*.swp
*.swo
""")

    print("\nDirectory structure setup complete!")
    print("\nReminder: Add your OpenAI API key to the .env file")
    print("You can now run the application with: python main.py")

if __name__ == "__main__":
    main()