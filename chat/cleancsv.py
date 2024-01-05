import csv

input_file = 'finetune_loss_7b.csv'
output_file = 'new_finetune_loss_7b.csv'

# Open the input file and the output file
with open(input_file, mode='r', newline='', encoding='utf-8') as infile, \
     open(output_file, mode='w', newline='', encoding='utf-8') as outfile:

    # Create CSV reader and writer
    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=[field for field in reader.fieldnames if field != 'Conversation History'])

    # Write the header without the 'Conversation History' column
    writer.writeheader()

    # Write the rows without the 'Conversation History' column
    for row in reader:
        del row['Conversation History']
        writer.writerow(row)

print("Column removed and new file saved as 'new_loss_13b.csv'")
