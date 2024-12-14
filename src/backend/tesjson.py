import json

# Path to your JSON file
json_file_path = "C:/Documents/a_new_journey_of_semester_3/Algeo/tubesAlgeo5/test/dataset/mapper/song.json"

# Read the JSON file
with open(json_file_path, "r") as file:
    albums_data = json.load(file)

# Print the data to verify
for album in albums_data:
    print(f"Album Name: {album['album_name']}")
    print(f"Artist Name: {album['artist_name']}")
    print(f"Song: {album['song']}")
    print(f"Image Path: {album['image_path']}")
    print("-" * 30)
