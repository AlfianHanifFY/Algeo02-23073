import os
import shutil

def cleanFolder(folder_path):
    if not os.path.exists(folder_path):
        return f"The folder '{folder_path}' does not exist."
    
    if not os.path.isdir(folder_path):
        return f"The path '{folder_path}' is not a directory."
    
    try:
        # Iterate over all files and directories in the folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # If it's a file, remove it
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            # If it's a directory, delete it and its contents
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        return f"All files and subdirectories in '{folder_path}' have been deleted."
    
    except Exception as e:
        return f"An error occurred while clearing the folder: {str(e)}"

