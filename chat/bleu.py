import csv
import sys
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

def bleu(input_csv, output_csv):
    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['BLEU Score']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        bleu_scores = []

        for row in reader:
            reference = word_tokenize(row['Ground Truth Answer'].lower())
            candidate = word_tokenize(row['Bot1 Response'].lower())
            score = sentence_bleu([reference], candidate)
            bleu_scores.append(score)
            row['BLEU Score'] = score
            writer.writerow(row)

        average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        writer.writerow({fieldnames[0]: 'Average BLEU Score', fieldnames[-1]: average_bleu})

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    bleu(input_csv, output_csv)


#python3 chat/bleu.py loss_7b.csv prompt_7b_bleu.csv
# python3 chat/bleu.py loss_13b.csv prompt_13b_bleu.csv

