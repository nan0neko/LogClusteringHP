import os
import csv

# Input file name (just the file name, not the full path)
input_file_name = 'samplelog.log'

# Open the input file using os.path.join to construct the full path
input_file_path = os.path.join(os.getcwd(), input_file_name)
with open(input_file_path, 'r') as log_file:

    # Read the contents of the log file
    log_lines = log_file.readlines()

    # Create a dictionary to hold the csv writers for each name
    csv_writers = {}

    # Loop through each line in the log file
    for line in log_lines:

        # Find the name in the line
        name = None
        if '- ' in line and ': ' in line:
            name = line.split('- ')[1].split(': ')[0]
        
        # If the name is /usr/sbin/cron[value1]/value2, set the name to 'value1'
        if name and '/usr/sbin/cron[' in name:
            name = 'usr'

        # If the name is value1[####] or value1[#####], set the name to 'value1'
        if name and '[' in name:
            name = name.split('[')[0]

        # If a name was found, create a csv writer for it if it doesn't exist
        if name:
            if name not in csv_writers:
                # Open a new file for writing the CSV data using os.path.join to construct the full path
                output_file_name = f"{name}.csv"
                output_file_path = os.path.join(os.getcwd(), output_file_name)
                csv_file = open(output_file_path, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writers[name] = (csv_file, csv_writer)

        # Write the line to the appropriate CSV file
        for csv_writer_name, csv_writer in csv_writers.items():
            if csv_writer_name == name:
                csv_writer[1].writerow([line.strip()])

# Close all the CSV files
for csv_file, csv_writer in csv_writers.values():
    csv_file.close()
