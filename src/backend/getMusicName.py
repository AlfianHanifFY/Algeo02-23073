import os

def get_music_filenames():
    # Navigate from the current script to the music directory
    directory = os.path.join(os.path.dirname(__file__), "../../test/dataset music")
    directory = os.path.abspath(directory)  # Get the absolute path for safety
    
    # Check if the directory exists
    if not os.path.exists(directory):
        return {"error": "Directory does not exist"}
    
    # List all files in the directory
    filenames = [file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    return filenames

file = get_music_filenames()

for i in file:
    print(i)


