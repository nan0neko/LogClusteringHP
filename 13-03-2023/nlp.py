import re
import csv

log_file = 'chassis.csv'
output_file = 'labeled_log.csv'

# Define regular expressions to extract required fields
timestamp_re = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
process_name_re = re.compile(r'processName=(\w+)')
process_id_re = re.compile(r'processId=(\d+)')
message_id_re = re.compile(r'messageId=(\d+)')
user_re = re.compile(r'user=(\w+)')
command_re = re.compile(r'command=(\w+)')

# Open input and output CSV files
with open(log_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
    reader = csv.reader(f_in)
    writer = csv.writer(f_out)

    # Write header row for output file
    writer.writerow(['Timestamp', 'Process Name', 'Process ID', 'Message ID', 'User', 'Command'])

    # Iterate over rows in input file and extract required fields
    for row in reader:
        message = row[0]

        timestamp_match = timestamp_re.search(message)
        if timestamp_match:
            timestamp = timestamp_match.group(1)
        else:
            timestamp = ''

        process_name_match = process_name_re.search(message)
        if process_name_match:
            process_name = process_name_match.group(1)
        else:
            process_name = ''

        process_id_match = process_id_re.search(message)
        if process_id_match:
            process_id = process_id_match.group(1)
        else:
            process_id = ''

        message_id_match = message_id_re.search(message)
        if message_id_match:
            message_id = message_id_match.group(1)
        else:
            message_id = ''

        user_match = user_re.search(message)
        if user_match:
            user = user_match.group(1)
        else:
            user = ''

        command_match = command_re.search(message)
        if command_match:
            command = command_match.group(1)
        else:
            command = ''

        # Write extracted fields to output file
        writer.writerow([timestamp, process_name, process_id, message_id, user, command])
