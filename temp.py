import os

# Get the list of all files in a directory
path = "plots/"
files = os.listdir(path)

# Print the files
for file in files:
    print(file)
