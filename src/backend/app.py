from flask import Flask, render_template, request, jsonify
import os
import zipfile

# Initialize Flask app
app = Flask(
    __name__, 
    static_folder="../frontend/static",  # Serve static files
    template_folder="../frontend"  # Serve HTML templates
)

# Configure file upload settings
UPLOAD_FOLDER_MUSIC = os.path.join(os.getcwd(), "test/dataset/music")
os.makedirs(UPLOAD_FOLDER_MUSIC, exist_ok=True)
UPLOAD_FOLDER_ALBUM = os.path.join(os.getcwd(), "test/dataset/album")
os.makedirs(UPLOAD_FOLDER_ALBUM, exist_ok=True)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "test/dataset/mapper")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER_MUSIC"] = UPLOAD_FOLDER_MUSIC
app.config["UPLOAD_FOLDER_ALBUM"] = UPLOAD_FOLDER_ALBUM
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  

@app.route("/")
def index():
    nama = ["alfian","heleni","soni"]
    return render_template("index.html", nama=nama)  

@app.route("/upload-dataset")
def uploadDataSet():
    return render_template("upload-dataset.html")



@app.route("/api/upload/audio-dataset", methods=["POST"])
def upload_dataset_music_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files[]")  # Get the list of uploaded files
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    uploaded_files = []

    for file in files:
        if file and file.filename != "":
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER_MUSIC, filename)

            # Save the uploaded file temporarily
            file.save(file_path)

            try:
                # Check if the uploaded file is a zip archive
                if zipfile.is_zipfile(file_path):
                    extract_folder_path = os.path.join(UPLOAD_FOLDER_MUSIC, os.path.splitext(filename)[0])
                    os.makedirs(extract_folder_path, exist_ok=True)

                    # Extract the contents
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder_path)

                    uploaded_files.append({
                        "original_file": filename,
                        "extracted_contents": os.listdir(extract_folder_path)
                    })

                    # Remove the original zip file
                    os.remove(file_path)

                else:
                    # If it's not a zip file, just save it as-is
                    uploaded_files.append({"file": filename})

            except Exception as e:
                return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    return jsonify({"message": "Files uploaded and extracted successfully", "uploaded_files": uploaded_files})

@app.route("/api/upload/album-dataset", methods=["POST"])
def upload_dataset_album_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files[]")  # Get the list of uploaded files
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400
    
    uploaded_files = []
    for file in files:
        if file and file.filename != "":
            file_path = os.path.join(app.config["UPLOAD_FOLDER_ALBUM"], file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
    
    return jsonify({"message": "Files uploaded successfully", "uploaded_files": uploaded_files})

@app.route("/api/upload/mapper-dataset", methods=["POST"])
def upload_dataset_mapper_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files[]")  # Get the list of uploaded files
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400
    
    uploaded_files = []
    for file in files:
        if file and file.filename != "":
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
    
    return jsonify({"message": "Files uploaded successfully", "uploaded_files": uploaded_files})


# Start the Flask server
if __name__ == "__main__":
    app.run(debug=True)  # Set debug=True for development
