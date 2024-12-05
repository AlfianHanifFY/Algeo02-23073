import librosa
import numpy as np
import mido
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine


def load_audio(file_path):
    y, sr = librosa.load(file_path)
    return y, sr


def extract_pitches(y, sr):
   
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    peaks, _ = find_peaks(onset_env, height=0.1)

    
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)

    detected_pitches = []
    for peak in peaks:
        index = np.argmax(magnitudes[:, peak]) 
        detected_pitches.append(pitches[index, peak])
    
    return detected_pitches, peaks / sr  


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
   
    y, sr = load_audio(audio_file)
    pitches, times = extract_pitches(y, sr)
    
    
    notes, timings = extract_midi_features(midi_file_to_compare)
    
    max_time_audio = max(times)
    normalized_times_audio = [t / max_time_audio for t in times]

    max_time_midi = max(timings)
    normalized_times_midi = [t / max_time_midi for t in timings]
    
    features_audio = np.array(list(zip(pitches, normalized_times_audio)))
    features_midi = np.array(list(zip(notes, normalized_times_midi)))
    
    features_audio_flat = features_audio.flatten()
    features_midi_flat = features_midi.flatten()
    
    features_audio_resampled, features_midi_resampled = resample_features(features_audio_flat, features_midi_flat)
    
    similarity = 1 - cosine(features_audio_resampled, features_midi_resampled)
    
    return similarity


midi_files = [
    "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/midi/PiratesoftheCaribbean-He'saPirate.mid",
    "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/midi/Bohemian-Rhapsody-1.mid",
    "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/midi/Never-Gonna-Give-You-Up.mid"
]


def search_manual_midi_files(audio_file, midi_files):
    similarities = {}
    for midi_file in midi_files:
        y, sr = load_audio(audio_file)
        pitches, times = extract_pitches(y, sr)
        
        notes, timings = extract_midi_features(midi_file)
        
        max_time_audio = max(times)
        normalized_times_audio = [t / max_time_audio for t in times]

        max_time_midi = max(timings)
        normalized_times_midi = [t / max_time_midi for t in timings]
        
        features_audio = np.array(list(zip(pitches, normalized_times_audio)))
        features_midi = np.array(list(zip(notes, normalized_times_midi)))
        
        features_audio_flat = features_audio.flatten()
        features_midi_flat = features_midi.flatten()
        
        features_audio_resampled, features_midi_resampled = resample_features(features_audio_flat, features_midi_flat)
        
        similarity = 1 - cosine(features_audio_resampled, features_midi_resampled)
        similarities[midi_file] = similarity
    
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_similarities

audio_file = "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/audio/soni.wav"

similarities = search_manual_midi_files(audio_file, midi_files)

for midi_file, similarity in similarities[:3]:
    print(f"MIDI File: {midi_file}, Similarity: {similarity}")