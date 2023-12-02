import psycopg2
from psycopg2 import sql


# Create table first.
# CREATE TABLE clinics
# (
#     "Clinic ID"      BIGSERIAL PRIMARY KEY,
#     "Clinic Name"    VARCHAR(255),
#     Latitude         DECIMAL(10, 6),
#     Longitude        DECIMAL(10, 6),
#     "Operation Time" VARCHAR(50)
# );
#
# CREATE TABLE Specialty
# (
#     "Specialty ID"   BIGSERIAL PRIMARY KEY,
#     "Clinic ID"    BIGINT,
#     "Specialty Name" TEXT
# );
#
#

def connect_to_db():
    try:
        # Connect to your postgres DB (Modify these with actual credentials)
        conn = psycopg2.connect(
            dbname='postgres',
            user='postgres',
            password='postgres',
            host='localhost',
            port='5432'
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


# Function to insert clinic data
def insert_clinic(conn, clinic_name, latitude, longitude, operation_time):
    with conn.cursor() as cur:
        cur.execute(sql.SQL(
            "INSERT INTO clinics (\"Clinic Name\", Latitude, Longitude, \"Operation Time\") VALUES (%s, %s, %s, %s) RETURNING \"Clinic ID\";"),
            (clinic_name, latitude, longitude, operation_time))
        clinic_id = cur.fetchone()[0]
        conn.commit()
        return clinic_id


# Function to insert specialty data
def insert_specialty(conn, clinic_id, specialty_name):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("INSERT INTO Specialty (\"Clinic ID\", \"Specialty Name\") VALUES (%s, %s);"),
                    (clinic_id, specialty_name))
        conn.commit()


# Main function to process the file and insert data
def process_and_insert_data(file_path):
    conn = connect_to_db()
    if conn is None:
        return

    with open(file_path, 'r') as file:
        clinic_data = file.read().split('\n\n')  # Splitting data by clinic

    for clinic_block in clinic_data:
        lines = clinic_block.split('\n')
        clinic_name = lines[0].split(': ')[1]
        specialties = lines[1].split(': ')[1].split(', ')
        latitude = float(lines[2].split(': ')[1])
        longitude = float(lines[3].split(': ')[1])
        operation_time = lines[4].split(': ')[1]

        # Insert clinic data and get the clinic ID
        clinic_id = insert_clinic(conn, clinic_name, latitude, longitude, operation_time)

        # Insert each specialty for the clinic
        for specialty in specialties:
            insert_specialty(conn, clinic_id, specialty)

    conn.close()


file_path = 'clinics1/clinic_data.txt'
process_and_insert_data(file_path)
