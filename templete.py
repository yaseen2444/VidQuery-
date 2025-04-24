import os

# Base directory
base_dir = "youtube-qa"

# File structure
file_structure = {
    "": ["app.py", "requirements.txt"],
    "utils": [
        "__init__.py",
        "transcribe.py",
        "embedding.py",
        "model.py",
        "retriever.py",
        "config.py"
    ]
}

# Placeholder content
placeholder_text = lambda name: f"# Placeholder for {name}\n"

# Create directories and files
for folder, files in file_structure.items():
    dir_path = os.path.join(base_dir, folder)
    os.makedirs(dir_path, exist_ok=True)
    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, "w") as f:
            f.write(placeholder_text(file))

print(f"✅ Project structure created at: {os.path.abspath(base_dir)}")
