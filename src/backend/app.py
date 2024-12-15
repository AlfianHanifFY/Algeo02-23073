from flask import Flask, render_template, request, jsonify,send_from_directory
import os
import zipfile
from API.loadJson import *
from jsonMaker import *
from cleanFolder import cleanFolder
from audioProcessing import *
from imageProcessing import *

# Initialize Flask app
app = Flask(
    __name__, 
     static_folder="../../test",  # Serve static files
    template_folder="../frontend"  # Serve HTML templates
)

# Configure file upload settings
UPLOAD_FOLDER_MUSIC = os.path.join(os.getcwd(), "test/dataset/music")
os.makedirs(UPLOAD_FOLDER_MUSIC, exist_ok=True)
UPLOAD_FOLDER_ALBUM = os.path.join(os.getcwd(), "test/dataset/album")
os.makedirs(UPLOAD_FOLDER_ALBUM, exist_ok=True)
UPLOAD_FOLDER = os.path.join(os.getcwd(), "test/dataset/mapper")
UPLOAD_QUERY_HUMMING = os.path.join(os.getcwd(), "test/query/humming")
UPLOAD_QUERY_IMAGE = os.path.join(os.getcwd(), "test/query/image")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER_MUSIC"] = UPLOAD_FOLDER_MUSIC
app.config["UPLOAD_FOLDER_ALBUM"] = UPLOAD_FOLDER_ALBUM
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["UPLOAD_QUERY_HUMMING"] = UPLOAD_QUERY_HUMMING
app.config["UPLOAD_QUERY_IMAGE"] = UPLOAD_QUERY_IMAGE
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  

@app.route("/")
def index():
    config = load_json("config/config.json")
    url = "dataset/mapper/" + config[0]['uploadedDatasetMapper']
    if config[0]['uploadedDatasetMapper'] == '':
        url = "dataset/mapper/awikwok"

    # Load all data
    data = load_json(url)

    # Pagination logic
    page = int(request.args.get('page', 1))  # Get page number from query string, default to 1
    per_page = 8  # Number of items per page
    total_pages = (len(data) + per_page - 1) // per_page  # Calculate total pages

    # Slice the data for the current page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data[start:end]

    return render_template(
        "home.html",
        data=paginated_data,
        config=config[0],
        current_page=page,
        total_pages=total_pages
    )
    
@app.route("/humming-result")
def humming_result_route():
    config = load_json("config/config.json")
    
    # create url
    mapperUrl = "test/dataset/mapper/" + config[0]['uploadedDatasetMapper']
    if config[0]['uploadedDatasetMapper'] == '':
        mapperUrl = "test/dataset/mapper/awikwok"
    audioUrl = "test/query/humming/" + config[0]['uploadedHumming']
    if config[0]['uploadedHumming'] == '':
        audioUrl = "test/query/humming/wwwww"
    
    # Load all data
    data,proccessing_time = search_manual_midi_files(audioUrl,mapperUrl,"test/dataset/music")

    # Pagination logic
    page = int(request.args.get('page', 1))  # Get page number from query string, default to 1
    per_page = 8  # Number of items per page
    total_pages = (len(data) + per_page - 1) // per_page  # Calculate total pages

    # Slice the data for the current page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data[start:end]

    return render_template(
        "humming-result.html",
        data=paginated_data,
        config=config[0],
        current_page=page,
        total_pages=total_pages,
        proccessing_time = proccessing_time
    )
    
@app.route("/album-result")
def album_result_route():
    config = load_json("config/config.json")
    
    # create url
    mapperUrl = "test/dataset/mapper/" + config[0]['uploadedDatasetMapper']
    if config[0]['uploadedDatasetMapper'] == '':
        mapperUrl = "test/dataset/mapper/awikwok"
    imageUrl = "test/query/image/" + config[0]['uploadedImage']
    if config[0]['uploadedImage'] == '':
        imageUrl = "test/query/image/wwwww"
    
    # Load all data
    data,proccessing_time = imageRetrieval(mapperUrl,"test/pca_result/", imageUrl)

    # Pagination logic
    page = int(request.args.get('page', 1))  # Get page number from query string, default to 1
    per_page = 8  # Number of items per page
    total_pages = (len(data) + per_page - 1) // per_page  # Calculate total pages

    # Slice the data for the current page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data[start:end]

    return render_template(
        "image-result.html",
        data=paginated_data,
        config=config[0],
        current_page=page,
        total_pages=total_pages,
        proccessing_time = proccessing_time
    )

@app.route("/upload-dataset")
def uploadDataSet():
    return render_template("upload-dataset.html")

@app.route("/api/upload/audio-dataset", methods=["POST"])
def upload_dataset_music_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("files[]")
    

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400
    updateConfig("uploadedDatasetMusic", files[0].filename)
    uploaded_files = []
    cleanFolder("test/dataset/music")
    for file in files:
        if file and file.filename != "":
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER_MUSIC, filename)

            # Save the uploaded file temporarily
            file.save(file_path)

            # Debug: Confirm file save
            if not os.path.exists(file_path):
                return jsonify({"error": f"File {file_path} was not saved"}), 500

            try:
                # Debug: Check if zipfile
                if zipfile.is_zipfile(file_path):
                    print(f"Extracting zip file: {file_path}")

                    extract_folder_path = os.path.join(UPLOAD_FOLDER_MUSIC)
                    os.makedirs(extract_folder_path, exist_ok=True)

                    # Extract contents
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder_path)

                    uploaded_files.append({
                        "original_file": filename,
                        "extracted_contents": os.listdir(extract_folder_path)
                    })

                    # Remove the original zip file
                    os.remove(file_path)
                else:
                    # Not a zip file
                    uploaded_files.append({"file": filename})

            except Exception as e:
                print(f"Error processing {filename}: {e}")
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
