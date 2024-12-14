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
def pca(X,k):
    # Mean centering data
    mean_pixel = np.mean(X, axis=0)
    X_centered= X - mean_pixel # Centering data

    #PCA dengan SVD
    U,Sigma,Vt = np.linalg.svd(X_centered, full_matrices=False)
    k = 4
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
        image_path = album["image_path"]  # Path gambar dari JSON
        
        processed_image = process_image(image_path)

        if processed_image is not None:
            images.append(processed_image)
            ids.append(album_id)
            
    X = np.array(images)

    # Hitung PCA
    mean_pixel, Uk, X_projected = pca(X, k=3)
    save_pca_results(mean_pixel, Uk, X_projected)

def main():
    current_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_path, '..', '..', 'test', 'dataset', 'mapper' , 'map.json')
    pca_result_path = os.path.join(current_path, '..', '..', 'test', 'pca_result', 'pca_results_mean_pixel.npy')

    with open(json_path, 'r') as file:
        album_data = json.load(file)

    start_time = time.time()
            
    #Jika file pca ada
    if not os.path.isfile(pca_result_path): 
        read_json_and_process(album_data) #klo udah diproses di awal, make as comment

    mean_pixel, Uk, X_projected = load_pca_results()
    
    dataset_folder_path_query = os.path.join(current_path, '..', '..', 'test', 'query')
    
    # Nama Query
    dataset_folder_path_query = os.path.join(current_path, '..', '..', 'test', 'query', 'image')
    query_image_input = input(f"Masukkan nama file gambar (.jpg) yang ada di folder {dataset_folder_path_query}: ")

    # Membentuk path lengkap ke file gambar
    image_file_path = os.path.join(dataset_folder_path_query, query_image_input)
    try:
        query_image = process_image(image_file_path)
    except Exception as e:
        print(f"Error memproses file lokal: {e}")
        query_image = None

    if query_image is not None:
        query_projected = projection_query_image(query_image, mean_pixel, Uk)

        #Jarak euclidean
        distances = np.linalg.norm(query_projected - X_projected, axis=1)
        
        sorted_indices = np.argsort(distances) #urutan index

        ordered_results = []
        for idx in sorted_indices:
            album_info = album_data[idx].copy()  # Salin informasi album
            album_info["distance"] = distances[idx]  # Tambahkan jarak ke informasi album
            ordered_results.append(album_info)  # Simpan informasi album ke hasil urutan

        end_time = time.time()
        processing_time = end_time - start_time
        return ordered_results, processing_time
    else:
        return None, None
'''
#try

if __name__ == "__main__":
    closest_image_id, process = main()
    print(f"Gambar: {closest_image_id}")
    print(f"Processing Time: {process}")
'''
