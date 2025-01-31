import os
import json
import librosa
import numpy as np
import mido
import time
from scipy.signal import find_peaks


def manual_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (norm_vec1 * norm_vec2)


# .wav
def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr

# .wav
def extract_pitches(y, sr):
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=0.1)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    detected_pitches = []
    for peak in peaks:
        index = np.argmax(magnitudes[:, peak])
        detected_pitches.append(pitches[index, peak])

    return detected_pitches, peaks / sr

# .midi
def extract_midi_features(midi_file_path):
    midi = mido.MidiFile(midi_file_path)
    notes = []
    timings = []

    for track in midi.tracks:
        current_time = 0
        for msg in track:
            if msg.type == 'note_on':
                notes.append(msg.note)
                timings.append(current_time)
            current_time += msg.time

    return notes, timings


def resample_features(features_1, features_2):
    len_1 = len(features_1)
    len_2 = len(features_2)

    if len_1 > len_2:
        new_features_1 = np.linspace(features_1[0], features_1[-1], len_2)
        return new_features_1, features_2
    elif len_1 < len_2:
        new_features_2 = np.linspace(features_2[0], features_2[-1], len_1)
        return features_1, new_features_2
    else:
        return features_1, features_2


def calculate_audio_midi_similarity(audio_file, midi_file_to_compare):
    # audio_file = query
    # midi_file_to_compare = dataset
    
    # load .wav
    y, sr = load_audio(audio_file)
    pitches, times = extract_pitches(y, sr)

    # load .mid
    notes, timings = extract_midi_features(midi_file_to_compare)

    # normalize
    max_time_audio = max(times)
    normalized_times_audio = [t / max_time_audio for t in times]

    max_time_midi = max(timings)
    normalized_times_midi = [t / max_time_midi for t in timings]

    features_audio = np.array(list(zip(pitches, normalized_times_audio)))
    features_midi = np.array(list(zip(notes, normalized_times_midi)))

    features_audio_flat = features_audio.flatten()
    features_midi_flat = features_midi.flatten()

    features_audio_resampled, features_midi_resampled = resample_features(features_audio_flat, features_midi_flat)

    # similarity
    similarity = manual_cosine_similarity(features_audio_resampled, features_midi_resampled)

    return similarity

def search_manual_midi_files(audio_file, json_file_path, base_dir):
    # Load MIDI file paths from the JSON file
    start_time = time.time()
    with open(json_file_path, "r") as json_file:
        midi_data = json.load(json_file)

    # initiate return
    similarities = []
    
    for entry in midi_data:
        midi_file = os.path.join(base_dir, entry["audio_file"])
        if not os.path.isfile(midi_file):
            print(f"File not found: {midi_file}")
            continue
        
        # load .wav
        y, sr = load_audio(audio_file)
        pitches, times = extract_pitches(y, sr)

        # load .mid ke-n
        notes, timings = extract_midi_features(midi_file)

        # normalize
        max_time_audio = max(times)
        normalized_times_audio = [t / max_time_audio for t in times]

        max_time_midi = max(timings)
        normalized_times_midi = [t / max_time_midi for t in timings]

        features_audio = np.array(list(zip(pitches, normalized_times_audio)))
        features_midi = np.array(list(zip(notes, normalized_times_midi)))

        features_audio_flat = features_audio.flatten()
        features_midi_flat = features_midi.flatten()

        features_audio_resampled, features_midi_resampled = resample_features(features_audio_flat, features_midi_flat)

        # similarity
        similarity = manual_cosine_similarity(features_audio_resampled, features_midi_resampled)
        
        # Append the result 
        similarities.append({
            "pic_name" : entry["pic_name"],
            "audio_file": entry["audio_file"],
            "similarity": similarity 
        })

    # Sort similarities descending
    sorted_similarities = sorted(similarities, key=lambda item: item["similarity"], reverse=True)
    end_time = time.time()
    processing_time = end_time - start_time
    return sorted_similarities, processing_time


# # Example usage
# audio_file = "/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/query/humming/ariana2.wav"
# json_file_path = "/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/dataset/mapper/final.json"
# base_dir = "/Users/alfianhaniffy/Documents/ITB/ALGEO/Algeo02-23073/test/dataset/music"

# similarities,proccessing_time = search_manual_midi_files(audio_file, json_file_path, base_dir)

# # Print top 4 results
# for result in similarities:
#     print(f"music: {result['audio_file']}, album: {result['pic_name']}, Similarity: {result['similarity']:.4f}, proses = {proccessing_time}")