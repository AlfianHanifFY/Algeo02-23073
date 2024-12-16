import mido
import numpy as np
import json
import librosa
import os
import time
from numpy.linalg import norm
from scipy.signal import find_peaks

def normalize_pitch(pitches, epsilon=1e-8):
    mean_pitch = np.mean(pitches)
    std_pitch = np.std(pitches)

    if std_pitch < epsilon:
        return np.zeros_like(pitches)  # Mengembalikan array nol jika variansi terlalu kecil

    res = (pitches - mean_pitch) / (std_pitch + epsilon)
    print(f"Mean pitch: {mean_pitch}, Std pitch: {std_pitch}")
    print(f"Normalized pitches: {res}")
    return res

def filter_pitches(pitches, threshold=50):
    """
    Menghapus pitch dengan nilai yang lebih kecil dari threshold.
    """
    return [pitch for pitch in pitches if pitch >= threshold]

def filter_sparse_notes(notes):
    return notes


def extract_pitches(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=0.1)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    detected_pitches = []
    for peak in peaks:
        index = np.argmax(magnitudes[:, peak])
        pitch_value = pitches[index, peak]
        print(f"Detected pitch at peak {peak}: {pitch_value}")

        if 100 < pitch_value < 1000:  
            detected_pitches.append(pitch_value)

    print(f"Total pitches detected: {len(detected_pitches)}")
    return detected_pitches, peaks / sr

def extract_pitches_and_chroma(y, sr):
    # Menggunakan metode pembuatan fitur pitch
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=0.1)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    detected_pitches = []
    for peak in peaks:
        index = np.argmax(magnitudes[:, peak])
        pitch_value = pitches[index, peak]
        if 100 < pitch_value < 1000:
            detected_pitches.append(pitch_value)

    print(f"Detected pitches: {detected_pitches}")

    # Menghitung fitur Chroma dari sinyal audio
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)  # Rata-rata setiap chroma bin

    return detected_pitches, chroma_mean


def process_wav(file_path, sr=22050, window_size=40, step_size=8):
    y, sr = librosa.load(file_path, sr=sr)
    
    detected_pitches, _ = extract_pitches(y, sr)
    detected_pitches = np.array(detected_pitches)

    # Validasi pitch non-zero dan tanpa NaN
    detected_pitches = detected_pitches[~np.isnan(detected_pitches)]
    detected_pitches = detected_pitches[detected_pitches > 0]
    
    normalized_pitches = normalize_pitch(detected_pitches)

    windows = [
        normalized_pitches[i:i + window_size]
        for i in range(0, len(normalized_pitches) - window_size + 1, step_size)
    ]

    if not windows:
        raise ValueError("Windowing produced no valid segments.")

    return windows
def process_midi(file_path, window_size=40, step_size=8):
    midi_file = mido.MidiFile(file_path)
    melody_notes = [
        msg.note for track in midi_file.tracks for msg in track
        if (msg.type == 'note_on')
    ]
    if not melody_notes:
        raise ValueError("No valid melody notes found in the MIDI file.")

    # Filter sparse notes
    filtered_notes = filter_sparse_notes(np.array(melody_notes))
    if len(filtered_notes) == 0:
        raise ValueError("Filtered notes are empty.")
    
    normalized_midi_notes = normalize_pitch(filtered_notes)
    windows = [
        normalized_midi_notes[i:i+window_size]
        for i in range(0, len(normalized_midi_notes) - window_size + 1, step_size)
    ]
    return windows

def compute_normalized_histogram(data, bins, range_):
    histogram, _ = np.histogram(data, bins=bins, range=range_)
    total = np.sum(histogram)

    if total == 0:
        return np.ones_like(histogram) / len(histogram)  # Distribusi uniform jika total nol
    
    normalized = histogram / np.sum(histogram)
    return normalized

def compute_atb(notes):
    return compute_normalized_histogram(notes, bins=np.arange(128), range_=(0, 127))

def compute_rtb(notes):
    intervals = np.diff(notes)
    return compute_normalized_histogram(intervals, bins=np.arange(-127, 127), range_=(-127, 127))

def compute_ftb(notes):
    intervals = notes - notes[0]
    return compute_normalized_histogram(intervals, bins=np.arange(-127, 127), range_=(-127, 127))

def extract_tone_distribution(notes):
    """
    Ekstraksi fitur distribusi tone berdasarkan ATB, RTB, dan FTB.
    """
    atb = compute_atb(notes)
    rtb = compute_rtb(notes)
    ftb = compute_ftb(notes)
    return {
        "ATB": atb,
        "RTB": rtb,
        "FTB": ftb
    }

def cosine_similarity(vec1, vec2):
    """
    Menghitung cosine similarity antara dua vektor.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = norm(vec1) * norm(vec2)
    if norm_product == 0:
        return 0
    return dot_product / norm_product

def compute_weighted_similarity(midi_distribution, wav_distribution, weights):
    similarities = {}
    for key in midi_distribution.keys():
        sim = cosine_similarity(midi_distribution[key], wav_distribution[key])
        similarities[key] = weights[key] * sim

    total_similarity = sum(similarities.values()) / sum(weights.values())
    return total_similarity


def find_similarities(query_file, json_file_path, base_dir, weights, window_size=20, step_size=8):
    start_time = time.time()

    try:
        with open(json_file_path, 'r') as file:
            mapper = json.load(file)
    except Exception as e:
        raise ValueError(f"Error loading JSON file: {e}")

    if query_file.endswith('.wav'):
        query_windows = process_wav(query_file, window_size=window_size, step_size=step_size)
        query_distribution = extract_tone_distribution(np.concatenate(query_windows))

    elif query_file.endswith('.mid'):
        query_windows = process_midi(query_file, window_size=window_size, step_size=step_size)
        query_distribution = extract_tone_distribution(np.concatenate(query_windows))

    else:
        raise ValueError("Unsupported query file type")

    results = []

    for file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, file)

        if file.endswith('.mid'):
            midi_windows = process_midi(file_path, window_size, step_size)
            midi_distribution = extract_tone_distribution(np.concatenate(midi_windows))

        elif file.endswith('.wav'):
            wav_windows = process_wav(file_path, window_size, step_size)
            wav_distribution = extract_tone_distribution(np.concatenate(wav_windows))

        similarity = compute_weighted_similarity(query_distribution, midi_distribution, weights)

        pic_name = next((entry['pic_name'] for entry in mapper if entry['audio_file'] == file), "Unknown")

        results.append({
            "audio_file": file,
            "pic_name": pic_name,
            "similarity": similarity
        })

    processing_time = time.time() - start_time
    return sorted(results, key=lambda x: x['similarity'], reverse=True), processing_time


# Example usage
# mapper_path = "/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/dataset/mapper/final.json"
# weights = {"ATB": 0.5, "RTB": 1.5, "FTB": 1}
# midi_folder = "/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/dataset/music"
# audio_file = "/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/query/humming/resonate 1734319883300.wav"

# results, time_taken = find_similarities(audio_file,mapper_path,midi_folder, weights)
# print("Results:", results)
# print("Processing time:", time_taken, "seconds")