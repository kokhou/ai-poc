import os

# First, let's read the content of the uploaded file to understand its structure and content.
file_path = '../clinics/clinic_data.txt'

with open(file_path, 'r') as file:
    clinic_data = file.read()

# Showing the first 500 characters to get an idea of the content and structure


# Splitting the content into different clinics
clinics = clinic_data.strip().split('\n\n')  # Splitting by double newline, assuming each clinic is separated this way

# Creating a directory to store the files
output_dir = '/clinics/'
os.makedirs(output_dir, exist_ok=True)


# Function to save each clinic's data into a separate file
def save_clinic_data(clinic, index):
    filename = f'clinic_{index}.txt'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as file:
        file.write(clinic)
    return filepath


# Saving each clinic's data
file_paths = [save_clinic_data(clinic, index) for index, clinic in enumerate(clinics)]

file_paths  # Displaying the paths for download links
