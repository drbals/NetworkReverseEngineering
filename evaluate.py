import sys

from model import inference


def evaluate(path_to_txt, path_to_true_idx_txt):  # ./data/protocol1/test.txt, ./data/protocol1/test-label.txt
    inference(path_to_txt)

    # Load true indices from the file, removing newline characters
    true_idx_list = []
    with open(path_to_true_idx_txt) as infile:
        for line in infile:
            true_idx_list.append(line.strip())

    # Load predicted indices from the file, removing newline characters
    predicted_idx_list = []
    with open(path_to_txt.replace(".txt", "-pred.txt")) as infile:
        for line in infile:
            predicted_idx_list.append(line.strip())

    # Each packet must have a predicted byte index for the message type
    assert len(true_idx_list) == len(predicted_idx_list), f"Incorrect format."

    # Calculate accuracy: count exact matches in order and position
    correct_predictions = sum(1 for true_idx, pred_idx in zip(true_idx_list, predicted_idx_list) if true_idx == pred_idx)
    accuracy = correct_predictions / len(true_idx_list) * 100

    print(f"Accuracy: {accuracy}%")

if __name__ == "__main__":
    evaluate(sys.argv[1], sys.argv[2])