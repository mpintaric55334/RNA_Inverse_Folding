import subprocess
from pathlib import Path
import torch


def save_position_matrix(name, matrix, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file = output_dir / f"{name}_original.pth"
    torch.save(matrix, file)


def load_position_matrix(name, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file = output_dir / f"{name}_original.pth"
    matrix = torch.load(file)
    return matrix


def metrics_calculation(original, predicted):

    total_ones = (original == 1).sum().item()
    true_positive = ((original == predicted) & (predicted == 1)).sum().item()
    false_positive = ((original != predicted) & (predicted == 1)).sum().item()
    false_negative = ((original != predicted) & (predicted == 0)).sum().item()

    return total_ones, true_positive, false_positive, false_negative


def save_sequence(sequence_tensor, name, output_dir):

    idx_to_nucl = {1: "A", 2: "C", 3: "G", 4: "U", 5: "N", 6: "P"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file = output_dir / f"{name}.fa"

    seq = ""

    for nucleotide in sequence_tensor:
        seq += idx_to_nucl[nucleotide.item()]

    with open(file, 'w') as f:
        f.write(f">{name}\n{seq}\n")


def save_sequence_true(sequence_tensor, name, output_dir):

    idx_to_nucl = {1: "A", 2: "C", 3: "G", 4: "U", 5: "N", 6: "P"}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file = output_dir / f"{name}_true.fa"

    seq = ""

    for nucleotide in sequence_tensor:
        seq += idx_to_nucl[nucleotide.item()]

    with open(file, 'w') as f:
        f.write(f">{name}\n{seq}\n")


def run_eternafold(name, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_file = output_dir / f"{name}.fa"
    bpseq_file = output_dir / f"{name}.bpseq"

    subprocess.check_call(
            [
                "eternafold", "predict",
                f"{seq_file}",
                "--bpseq", f"{bpseq_file}",

            ],

            stdout=subprocess.DEVNULL,

            stderr=subprocess.STDOUT,

        )


def bpseq_to_matrix(name, output_dir):

    output_dir = Path(output_dir)
    bpseq_file = output_dir / f"{name}.bpseq"
    lines = []

    with open(bpseq_file, "r") as f:
        for line in f:
            if len(line.split(" ")) == 3:
                lines.append(line.strip())

        matrix = torch.zeros((len(lines), len(lines)), device="cuda")
        for line in lines:
            pair_one, _, pair_two = line.split(" ")
            if int(pair_two) != 0:
                matrix[int(pair_one)-1, int(pair_two)-1] = 1
 
    return matrix
