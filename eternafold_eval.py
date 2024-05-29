from utils.folding import load_position_matrix, save_position_matrix, metrics_calculation, bpseq_to_matrix, run_eternafold


total_, tp_, fp_, fn_ = 0, 0, 0, 0
for name in range(2517):

    run_eternafold(str(name), "output_dir")
    original = load_position_matrix(str(name), "output_dir")
    reconstructed = bpseq_to_matrix(str(name), "output_dir")
    total, tp, fp, fn = metrics_calculation(original, reconstructed)
    total_ += total
    tp_ += tp
    fp_ += fp
    fn_ += fn

print(total_, tp_, fp_, fn_)

