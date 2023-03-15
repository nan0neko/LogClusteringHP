import spacy
import csv

log_file = 'chassis.csv'
output_file = 'labeled_log.csv'

# Load pre-trained spaCy model with text classification pipeline
nlp = spacy.load('en_core_web_sm')

# Define labels for text classification
label_names = ['Timestamp', 'Process Name', 'Process ID', 'Message ID', 'User', 'Command']

# Define function to extract required fields from log message using spaCy
def extract_fields(message):
    doc = nlp(message)

    timestamp = ''
    process_name = ''
    process_id = ''
    message_id = ''
    user = ''
    command = ''

    # Iterate over entities in the spaCy document and extract required fields
    for ent in doc.ents:
        if ent.label_ == 'Timestamp':
            timestamp = ent.text
        elif ent.label_ == 'Process Name':
            process_name = ent.text
        elif ent.label_ == 'Process ID':
            process_id = ent.text
        elif ent.label_ == 'Message ID':
            message_id = ent.text
        elif ent.label_ == 'User':
            user = ent.text
        elif ent.label_ == 'Command':
            command = ent.text

    return [timestamp, process_name, process_id, message_id, user, command]

# Open input and output CSV files
with open(log_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Write header row for output file
    writer.writerow(['Timestamp', 'Process Name', 'Process ID', 'Message ID', 'User', 'Command'])

    # Iterate over rows in input file and extract required fields
    for row in reader:
        message = row[0]

        # Apply text classification and extract required fields
        fields = extract_fields(message)

        # Print extracted fields
        print(fields)

        # Write extracted fields to output file
        writer.writerow(fields)
print(reader)