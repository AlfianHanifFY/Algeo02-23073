import numpy as np
import os
import matplotlib.pyplot as plt
#from kaggle.api.kaggle_api_extended import KaggleApi
from io import BytesIO
import requests
import pandas as pd
from PIL import Image

#os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\ASUS\.kaggle'

#api = KaggleApi()
#api.authenticate()

#dataset_name = 'receplyasolu/6k-weather-labeled-spotify-songs'  # Nama Dataset
download_path = 'datasets\spotify_weather'  # Path to save

#api.dataset_download_files(dataset_name, path=download_path, unzip=True)
#print(f"Dataset berhasil diunduh dan disimpan di: {download_path}")

#proses image -> gray -> resize -> flatten
def process_image(image_path_or_url, target_size = (28,28)):
    try: 
        if image_path_or_url.startswith("http://") or image_path_or_url.startswith("https://"):
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
        else:
            image = Image.open(image_path_or_url)
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
        print(f"Error memproses gambar dari URL: {image_url}. Error: {e}")
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


#Load CSV file
current_path = os.getcwd()
relative_path = os.path.join("src", "backend", "datasets","spotify_weather")
dataset_folder_path = os.path.join(current_path, relative_path)

csv_file = input("Masukkan path file CSV dataset: ")
image_file_path = os.path.join(dataset_folder_path, csv_file)
if os.path.isfile(image_file_path):
    print(f"File ditemukan: {image_file_path}")
    data = pd.read_csv(image_file_path)

    image_urls = data['Image']

    images = []
    image_info = []  # Menyimpan nama track/artis
    for idx, image_url in enumerate(image_urls):
        processed_image = process_image(image_url)
        if processed_image is not None:
            images.append(processed_image)
            image_info.append(f"{data.loc[idx, 'Artist']} - {data.loc[idx, 'Album']}")
    X = np.array(images)

    # Hitung PCA secara manual
    mean_pixel, Uk, X_projected = pca(X, k=3)

    #Query dengan URL/Path file lokal
    query_image_input = input("Masukkan URL gambar / path file lokal yang ingin dicek: ")
    if query_image_input.startswith("http://") or query_image_input.startswith("https://"):
        query_image = process_image(query_image_input)
    else:
        relative_path_query = os.path.join("src", "backend","query")
        dataset_folder_path_query = os.path.join(current_path, relative_path_query)

        image_file_path = os.path.join(dataset_folder_path_query, query_image_input)
        if not os.path.isfile(image_file_path):
            print(f"File {image_file_path} tidak ditemukan.")
            query_image = None
        elif not query_image_input.lower().endswith(".jpg"):
            print("File bukan berupa .jpg.")
            query_image = None
        else:
            try:
                image = Image.open(image_file_path)
                query_image = process_image(image_file_path)
            except Exception as e:
                print(f"Error memproses file lokal: {e}")
                query_image = None

    if query_image is None:
        print("Gagal memproses gambar query.")
    else:
        query_projected = projection_query_image(query_image, mean_pixel, Uk)

        #Jarak euclidean
        distances = np.zeros(X_projected.shape[0])  # Array simpen jarak euclidean

        for i in range(X_projected.shape[0]):
            # kurangkan + kuadrat
            squared_diff = (query_projected - X_projected[i]) ** 2
            # sum + akar kuadrat
            distances[i] = np.sqrt(np.sum(squared_diff))

        #Urutkan gambar
        sorted_indices = np.argsort(distances) #semakin jarak kecil, semakin mirip

        #Ambil 3 gambar teratas (contoh)
        num_results = 3
        top_k_images = sorted_indices[:num_results]

        #Tampilkan hasil gambar (just to check)
        print("Gambar yang paling mirip dengan query:")
        for idx in top_k_images:
            #print(f"File gambar: {image_files[idx]}, Jarak Euclidean: {distances[idx]}")
            print(f"Track: {image_info[idx]}, Jarak Euclidean: {distances[idx]}")

        for idx in top_k_images:
            image_url = image_urls[idx]
            print(f"Menampilkan gambar: {image_info[idx]} (URL: {image_url})")
            
            try:
                # Download gambar
                response = requests.get(image_url)
                response.raise_for_status()
                image_data = BytesIO(response.content)
                # Load gambar menggunakan PIL
                image = Image.open(image_data)

                # Tampilkan gambar use Matplotlib
                plt.figure()
                plt.imshow(image)
                plt.title(image_info[idx])
                plt.axis('off')
            except Exception as e:
                print(f"Error menampilkan gambar dari URL: {image_url}. Error: {e}")

        plt.show()
else:
    print(f"File '{csv_file}' tidak ditemukan di {image_file_path}")