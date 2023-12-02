import pandas as pd
import re
from sqlalchemy import create_engine
import urllib.parse

# Database credentials
hostname = 'localhost'
port = 3306  # default MySQL port
username = 'root'
password = urllib.parse.quote('P@ssw0rd')
database_name = 'poc'

# Create a connection string
connection_str = f'mysql+pymysql://{username}:{password}@{hostname}:{port}/{database_name}'

# Create an engine to connect to the MySQL server
engine = create_engine(connection_str)

# Let's first read the content of the uploaded file to understand its structure and data.
file_path = '../clinics1/clinic_data.txt'

with open(file_path, 'r') as file:
    clinic_data = file.read()

# Splitting the data into individual clinic entries
clinic_entries = clinic_data.strip().split('\n\n')

# Extracting and organizing the data into the proposed structure
clinic_list = []
service_list = []

# Unique IDs for clinic and services
clinic_id = 1
service_id = 1

for entry in clinic_entries:
    # Splitting each entry into lines for easy processing
    lines = entry.split('\n')

    # Extracting clinic data
    clinic_name = re.search(r'Clinic Name: (.+)', lines[0]).group(1)
    services = re.search(r'Services: (.+)', lines[1]).group(1).split(', ')
    latitude = float(re.search(r'Location Latitude: (.+)', lines[2]).group(1))
    longitude = float(re.search(r'Location Longitude: (.+)', lines[3]).group(1))
    operation_time = re.search(r'Operation Time: (.+)', lines[4]).group(1)

    # Adding clinic data to the list
    clinic_list.append([clinic_id, clinic_name, latitude, longitude, operation_time])

    # Adding service data to the list
    for service in services:
        service_list.append([service_id, clinic_id, service])
        service_id += 1

    # Incrementing the clinic_id for the next clinic
    clinic_id += 1

# Converting lists to DataFrames
clinics_df = pd.DataFrame(clinic_list, columns=['Clinic ID', 'Clinic Name', 'Latitude', 'Longitude', 'Operation Time'])
services_df = pd.DataFrame(service_list, columns=['Service ID', 'Clinic ID', 'Service Name'])

# clinics_df.head(), services_df.head()

# Assuming you have a DataFrame named 'clinics_df'
clinics_df.to_sql('clinics', con=engine, if_exists='replace', index=False)

# And for the services DataFrame
services_df.to_sql('services', con=engine, if_exists='replace', index=False)

