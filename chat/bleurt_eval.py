import csv
import sys
import os
from bleurt import score

def bleurt(input_csv, output_csv):
    checkpoint = "bleurt/bleurt/test_checkpoint"
    scorer = score.BleurtScorer(checkpoint)

    with open(input_csv, newline='', encoding='utf-8') as infile, \
         open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['BLEURT Score']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        bleurt_scores = []

        for row in reader:
            reference = [row['Ground Truth Answer']]
            candidate = [row['Bot1 Response']]
            scores = scorer.score(references=reference, candidates=candidate)
            bleurt_score = scores[0]
            bleurt_scores.append(bleurt_score)
            row['BLEURT Score'] = bleurt_score
            writer.writerow(row)

        # Calculate and write average BLEURT score
        print(print('bluert_score',  bleurt_scores))
        print('length of bluert_score',  len(bleurt_scores))
        print('sum of bluert_score',  sum(bleurt_scores))
        average_bleurt_score = sum(bleurt_scores) / len(bleurt_scores) if bleurt_scores else 0
        writer.writerow({fieldnames[0]: 'Average BLEURT Score', fieldnames[-1]: average_bleurt_score})

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv_file> <output_csv_file>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    bleurt(input_csv, output_csv)


#python3 chat/bleurt_eval.py loss_7b.csv prompt_7b_bleurt.csv
#python3 chat/bleurt_eval.py loss_13b.csv prompt_13b_bleurt.csv

#python3 chat/bleurt_eval.py finetune_loss_7b.csv prompt_7b_bleurt.csv

# python3 chat/bleurt_eval.py new_finetune_loss_7b.csv prompt_finetune_7b_bleurt.csv
# python3 chat/bleu.py new_finetune_loss_7b.csv prompt_finetune_7b_bleu.csv