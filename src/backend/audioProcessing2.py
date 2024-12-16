import os
import json
import librosa
import numpy as np
import mido
from scipy.signal import find_peaks

def manual_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0  # Avoid division by zero
    return dot_product / (norm_vec1 * norm_vec2)


def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def extract_audio_features(file_path):
    y, sr = load_audio(file_path)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=0.1)
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    detected_pitches = []
    for peak in peaks:
        index = np.argmax(magnitudes[:, peak])
        detected_pitches.append(pitches[index, peak])

    times = peaks / sr
    max_time = max(times, default=1)
    normalized_times = [t / max_time for t in times]

    return np.array(list(zip(detected_pitches, normalized_times))).flatten()


def extract_midi_features(midi_file_path):
    midi = mido.MidiFile(midi_file_path)
    notes = []
    timings = []

    for track in midi.tracks:
        current_time = 0
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
                timings.append(current_time)
            current_time += msg.time

    max_time = max(timings, default=1)
    normalized_times = [t / max_time for t in timings]

    return np.array(list(zip(notes, normalized_times))).flatten()


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


def calculate_similarity(file1, file2):
    # Determine if the files are audio or MIDI
    is_file1_midi = file1.lower().endswith(".mid") or file1.lower().endswith(".midi")
    is_file2_midi = file2.lower().endswith(".mid") or file2.lower().endswith(".midi")

    features1 = extract_midi_features(file1) if is_file1_midi else extract_audio_features(file1)
    features2 = extract_midi_features(file2) if is_file2_midi else extract_audio_features(file2)

    # Resample features to ensure equal length
    features1_resampled, features2_resampled = resample_features(features1, features2)

    # Calculate cosine similarity
    similarity = manual_cosine_similarity(features1_resampled, features2_resampled)

    return similarity


def search_manual_files(audio_or_midi_file, json_file_path, base_dir):
    # Load MIDI or WAV file paths from the JSON file
    with open(json_file_path, "r") as json_file:
        file_data = json.load(json_file)

    similarities = []
    for entry in file_data:
        target_file = os.path.join(base_dir, entry["song"])
        if not os.path.isfile(target_file):
            print(f"File not found: {target_file}")
            continue

        similarity = calculate_similarity(audio_or_midi_file, target_file)
        
        # Append the result with id, song, and similarity
        similarities.append({
            "id": entry["id"],
            "song": entry["song"],
            "similarity": similarity
        })

    # Sort similarities by similarity value in descending order
    sorted_similarities = sorted(similarities, key=lambda item: item["similarity"], reverse=True)

    return sorted_similarities


# Example usage
audio_or_midi_file = "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/audio/ariana2.wav"
json_file_path = "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/dataset/mapper/wav.json"
base_dir = "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5"

similarities = search_manual_files(audio_or_midi_file, json_file_path, base_dir)

# Print top 4 results
for result in similarities[:4]:
    print(f"ID: {result['id']}, Song: {result['song']}, Similarity: {result['similarity']:.4f}")

