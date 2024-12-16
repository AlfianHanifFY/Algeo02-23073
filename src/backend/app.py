from flask import Flask, render_template, request, jsonify,send_from_directory
import os
import zipfile
from API.loadJson import *
from jsonMaker import *
from cleanFolder import cleanFolder
from imageProcessing import *
from newAudio import *


app = Flask(
    __name__, 
     static_folder="../../test",  
    template_folder="../frontend" 
)


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
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024



@app.route("/")
def index():
    config = load_json("config/config.json")
    url = "dataset/mapper/" + config[0]['uploadedDatasetMapper']
    if config[0]['uploadedDatasetMapper'] == '':
        url = "dataset/mapper/awikwok"

   
    data = load_json(url)


    page = int(request.args.get('page', 1))  
    per_page = 8  
    total_pages = (len(data) + per_page - 1) // per_page  


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
    weights = {"ATB": 0.5, "RTB": 2, "FTB": 1 }
    
    mapperUrl = "test/dataset/mapper/" + config[0]['uploadedDatasetMapper']
    if config[0]['uploadedDatasetMapper'] == '':
        mapperUrl = "test/dataset/mapper/awikwok"
    audioUrl = "test/query/humming/" + config[0]['uploadedHumming']
    if config[0]['uploadedHumming'] == '':
        audioUrl = "test/query/humming/wwwww"
    
   
    data,proccessing_time = find_similarities(audioUrl,mapperUrl,"test/dataset/music/",weights)

   
    page = int(request.args.get('page', 1))  
    per_page = 8 
    total_pages = (len(data) + per_page - 1) // per_page  


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
    

    mapperUrl = "test/dataset/mapper/" + config[0]['uploadedDatasetMapper']
    if config[0]['uploadedDatasetMapper'] == '':
        mapperUrl = "test/dataset/mapper/awikwok"
    imageUrl = "test/query/image/" + config[0]['uploadedImage']
    if config[0]['uploadedImage'] == '':
        imageUrl = "test/query/image/wwwww"
    

    data,proccessing_time = imageRetrieval(mapperUrl,"test/pca_result/", imageUrl)


    page = int(request.args.get('page', 1))  
    per_page = 8  
    total_pages = (len(data) + per_page - 1) // per_page  


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


            file.save(file_path)


            if not os.path.exists(file_path):
                return jsonify({"error": f"File {file_path} was not saved"}), 500

            try:

                if zipfile.is_zipfile(file_path):
                    print(f"Extracting zip file: {file_path}")

                    extract_folder_path = os.path.join(UPLOAD_FOLDER_MUSIC)
                    os.makedirs(extract_folder_path, exist_ok=True)


                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder_path)

                    uploaded_files.append({
                        "original_file": filename,
                        "extracted_contents": os.listdir(extract_folder_path)
                    })


                    os.remove(file_path)
                else:

                    uploaded_files.append({"file": filename})

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    return jsonify({"message": "Files uploaded and extracted successfully", "uploaded_files": uploaded_files})


@app.route("/api/upload/album-dataset", methods=["POST"])
def upload_dataset_album_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files[]")  
    updateConfig("uploadedDatasetAlbum", files[0].filename)
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400
    
    uploaded_files = []
    cleanFolder("test/dataset/album")
    for file in files:
        if file and file.filename != "":
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER_ALBUM, filename)

            
            file.save(file_path)

           
            if not os.path.exists(file_path):
                return jsonify({"error": f"File {file_path} was not saved"}), 500

            try:
               
                if zipfile.is_zipfile(file_path):
                    print(f"Extracting zip file: {file_path}")

                    extract_folder_path = os.path.join(UPLOAD_FOLDER_ALBUM)
                    os.makedirs(extract_folder_path, exist_ok=True)

                   
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder_path)

                    uploaded_files.append({
                        "original_file": filename,
                        "extracted_contents": os.listdir(extract_folder_path)
                    })

                    
                    os.remove(file_path)
                else:
                   
                    uploaded_files.append({"file": filename})

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

    return jsonify({"message": "Files uploaded and extracted successfully", "uploaded_files": uploaded_files})

@app.route("/api/upload/mapper-dataset", methods=["POST"])
def upload_dataset_mapper_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files[]") 
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    allowed_extensions = {".json"}
    uploaded_files = []
    cleanFolder("test/dataset/mapper")
    cleanFolder("test/pca_result")
    for file in files:
        if file and file.filename != "":

            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in allowed_extensions:
                return jsonify({"error": f"Invalid file type: {file.filename}. Only .json files are allowed."}), 400
            
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
    
    updateConfig("uploadedDatasetMapper", uploaded_files[0])
    return jsonify({"message": "Files uploaded successfully", "uploaded_files": uploaded_files})

@app.route("/api/upload/humming-query", methods=["POST"])
def upload_query_humming_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files[]")  
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    allowed_extensions = {".wav",".mid"}
    uploaded_files = []
    cleanFolder("test/query/humming")
    for file in files:
        if file and file.filename != "":
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in allowed_extensions:
                return jsonify({"error": f"Invalid file type: {file.filename}. Only .wav or .mid files are allowed."}), 400
            
            file_path = os.path.join(app.config["UPLOAD_QUERY_HUMMING"], file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
    updateConfig("uploadedHumming", uploaded_files[0])
    return jsonify({"message": "Files uploaded successfully", "uploaded_files": uploaded_files})

@app.route("/api/upload/image-query", methods=["POST"])
def upload_query_image_files():
    if "files[]" not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist("files[]") 
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No selected files"}), 400

    allowed_extensions = {".png",".jpeg",".jpg"}
    uploaded_files = []
    cleanFolder("test/query/image")
    for file in files:
        if file and file.filename != "":
            _, file_extension = os.path.splitext(file.filename)
            if file_extension.lower() not in allowed_extensions:
                return jsonify({"error": f"Invalid file type: {file.filename}. Only .png, .jpeg, .jpg files are allowed."}), 400
            
            file_path = os.path.join(app.config["UPLOAD_QUERY_IMAGE"], file.filename)
            file.save(file_path)
            uploaded_files.append(file.filename)
    updateConfig("uploadedImage", uploaded_files[0])
    return jsonify({"message": "Files uploaded successfully", "uploaded_files": uploaded_files})




if __name__ == "__main__":
    app.run(debug=True)  
