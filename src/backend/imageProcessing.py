import numpy as np
import os
from PIL import Image
import json
import time

#proses image -> gray -> resize -> flatten
def process_image(image_path, target_size = (64,64)): #cek ukuran
    try: 
        image = Image.open(image_path)

        #grayscale
        image_gray = np.array(image)
        if image_gray.ndim == 3:
            image_gray = np.dot(image_gray[...,:3], [0.2989, 0.5870, 0.1140])

        #resized
        height, width = image_gray.shape
        target_height, target_width = target_size
        resized_image = np.zeros((target_height, target_width))

        for i in range(target_height):
            for j in range(target_width):
                src_i = int(i * height / target_height)
                src_j = int(j * width / target_width)
                resized_image[i, j] = image_gray[src_i, src_j]

        #flatten gambar
        flattened_image = resized_image.flatten()
        return flattened_image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

#PCA
def pca(X, variance_threshold=0.95):
    # Mean centering data
    mean_pixel = np.mean(X, axis=0)
    X_centered= X - mean_pixel # Centering data

    #PCA dengan SVD
    U,Sigma,Vt = np.linalg.svd(X_centered, full_matrices=False)

    cumulative_variance = np.cumsum(Sigma**2) / np.sum(Sigma**2)
    k = np.argmax(cumulative_variance >= variance_threshold) + 1
    
    Uk = Vt.T[:,:k]
    X_projected = X_centered.dot(Uk)

    return mean_pixel, Uk, X_projected

# Memproyeksikan gambar query ke ruang komponen utama
def projection_query_image(query_image, mean_pixel, Uk):
    query_image_centered = query_image - mean_pixel  # Centering gambar query
    query_projected = query_image_centered.dot(Uk)  # Proyeksi gambar query ke ruang PCA
    return query_projected

# Fungsi untuk menyimpan hasil PCA
def save_pca_results(mean_pixel, Uk, X_projected, prefix="pca_results"):
    current_path = os.path.dirname(os.path.abspath(__file__))
    result_folder = os.path.join(current_path, '..', '..', 'test', 'pca_result')
    os.makedirs(result_folder, exist_ok=True)

    np.save(os.path.join(result_folder, f"{prefix}_mean_pixel.npy"), mean_pixel)
    np.save(os.path.join(result_folder, f"{prefix}_Uk.npy"), Uk)
    np.save(os.path.join(result_folder, f"{prefix}_X_projected.npy"), X_projected)

# Fungsi untuk memuat hasil PCA
def load_pca_results(prefix="pca_results"):
    result_folder = os.path.join('test', 'pca_result')
    
    mean_pixel = np.load(os.path.join(result_folder, f"{prefix}_mean_pixel.npy"))
    Uk = np.load(os.path.join(result_folder, f"{prefix}_Uk.npy"))
    X_projected = np.load(os.path.join(result_folder, f"{prefix}_X_projected.npy"))
    return mean_pixel, Uk, X_projected

# Read JSON + Process + Numpy Array + PCA
def read_json_and_process (album_data):
    images = []
    ids = []
    for idx, album in enumerate (album_data):
        album_id = album.get("id", f"{idx}")
        album["id"] = album_id  # Tambahkan ID ke file JSON
        image_path = "test/dataset/album/" + album["pic_name"]  # Path gambar dari JSON
        
        processed_image = process_image(image_path)

        if processed_image is not None:
            images.append(processed_image)
            ids.append(album_id)
            
    X = np.array(images)

    # Hitung PCA
    mean_pixel, Uk, X_projected = pca(X, variance_threshold=0.95)
    save_pca_results(mean_pixel, Uk, X_projected)

def imageRetrieval(json_path,pca_result_path,query_path):
    with open(json_path, 'r') as file:
        album_data = json.load(file)
            
    #Jika file pca ada
    if not os.path.isfile(pca_result_path): 
        read_json_and_process(album_data) #klo udah diproses di awal, make as comment

    mean_pixel, Uk, X_projected = load_pca_results()

    # Membentuk path lengkap ke file gambar
    start_time = time.time()
    image_file_path = query_path
    try:
        query_image = process_image(image_file_path)
    except Exception as e:
        print(f"Error memproses file lokal: {e}")
        query_image = None

    if query_image is not None:
        query_projected = projection_query_image(query_image, mean_pixel, Uk)

        #Jarak euclidean
        distances = np.linalg.norm(query_projected - X_projected, axis=1)
        
        # Jarak maksimum 
        max_distance = np.max(distances)

        sorted_indices = np.argsort(distances) #urutan index
        
        ordered_results = []
        for idx in sorted_indices:
            album_info = album_data[idx].copy()  # Salin informasi album
            album_info["distance"] = distances[idx]  # Tambahkan jarak ke informasi album

            similarity_percentage = (1 - (distances[idx] / max_distance)) * 100
            similarity_percentage = max(0.0, similarity_percentage)  # Menghindari nilai negatif

            album_info["similarity"] = similarity_percentage  # Tambahkan persentase kemiripan
            ordered_results.append(album_info)  # Simpan informasi album ke hasil urutan

        end_time = time.time()
        processing_time = end_time - start_time
        return ordered_results, processing_time
    else:
        return None, None

#try

# if __name__ == "__main__":
#     closest_image_id, process = imageRetrieval("/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/dataset/mapper/final.json","/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/pca_result","/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/query/image/Screenshot 2024-12-15 at 23.29.12.png")
#     print(f"Gambar: {closest_image_id}")
#     print(f"Processing Time: {process}")

