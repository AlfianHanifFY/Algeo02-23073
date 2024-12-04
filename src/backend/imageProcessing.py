import numpy as np
import os
from PIL import Image
import json

#proses image -> gray -> resize -> flatten
def process_image(image_path, target_size = (30,30)):
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
    n_samples, n_features = X.shape
    mean_pixel = np.sum(X, axis=0) / n_samples  # Menghitung rata-rata manual
    X_centered = X - mean_pixel  # Centering data
    
    # Hitung matriks kovarians 
    covariance_matrix = np.zeros((n_features, n_features))
    for i in range(n_features):
        for j in range(n_features):
            covariance_matrix[i, j] = np.sum(X_centered[:, i] * X_centered[:, j]) / (n_samples - 1)
    # Eigen decomposition 
    def power_iteration(matrix, num_simulations=1000, tolerance=1e-6):
        b_k = np.random.rand(matrix.shape[1])  # Vektor awal acak -> perkiraan awal untuk eigenvektor
        for _ in range(num_simulations):
            # Hitung hasil perkalian matriks dan vektor
            b_k1 = np.dot(matrix, b_k)
            # Normalisasi vektor
            b_k1_norm = np.linalg.norm(b_k1)
            b_k = b_k1 / b_k1_norm
            # Cek konvergensi
            if np.allclose(matrix @ b_k, b_k1_norm * b_k, atol=tolerance):
                break
        eigenvalue = b_k1_norm
        eigenvector = b_k
        return eigenvalue, eigenvector
    
    # Iterasi untuk menemukan eigenvalues dan eigenvectors
    eigvals = []
    eigvecs = []
    A = covariance_matrix.copy()
    for _ in range(k):  # Ambil k komponen utama
        eigenvalue, eigenvector = power_iteration(A)
        eigvals.append(eigenvalue)
        eigvecs.append(eigenvector)
        # Reduksi matriks untuk mencari eigenvalue/eigenvector berikutnya
        A = A - eigenvalue * np.outer(eigenvector, eigenvector)
    
    eigvals = np.array(eigvals)
    eigvecs = np.array(eigvecs).T  # Transpose agar setiap kolom adalah eigenvector
    
    # Proyeksikan data ke ruang PCA
    Uk = eigvecs[:, :k]
    X_projected = X_centered.dot(Uk)
    
    return mean_pixel, Uk, X_projected

# Memproyeksikan gambar query ke ruang komponen utama
def projection_query_image(query_image, mean_pixel, Uk):
    query_image_centered = query_image - mean_pixel  # Centering gambar query
    query_projected = query_image_centered.dot(Uk)  # Proyeksi gambar query ke ruang PCA
    return query_projected

# Fungsi untuk menyimpan hasil PCA
def save_pca_results(mean_pixel, Uk, X_projected, prefix="pca_results"):
    result_folder = os.path.join('test', 'pca_result')
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
    for album in album_data:
        album_id = album["id"]  # ID dari file JSON
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
    json_path = os.path.join(current_path, '..', '..', 'test', 'map.json')

    with open(json_path, 'r') as file:
        album_data = json.load(file)

    read_json_and_process(album_data) #klo udah diproses di awal, make as comment

    mean_pixel, Uk, X_projected = load_pca_results()

    dataset_folder_path_query = r'D:\ImageProcessing\Algeo2\Algeo02-23073\test\query'

    # Nama Query
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
        distances = np.zeros(X_projected.shape[0])  # Array simpen jarak euclidean

        for i in range(X_projected.shape[0]):
            # kurangkan + kuadrat
            squared_diff = (query_projected - X_projected[i]) ** 2
            # sum + akar kuadrat
            distances[i] = np.sqrt(np.sum(squared_diff))
        
        sorted_indices = np.argsort(distances) #urutan index
        closest_image_idx = sorted_indices[0] # 1 gambar dengan jarak paling kecil

        closest_image_id = album_data[closest_image_idx]["id"]
        return closest_image_id, distances[closest_image_idx]
    else:
        return None, None

#try

if __name__ == "__main__":
    closest_image_id, closest_distance = main()
    print(f"ID Gambar Terdekat: {closest_image_id}")
    print(f"Jarak Euclidean: {closest_distance}")
