"""Minimal example: predict a single miRNA-target interaction."""

from insect_mirna_target import predict

probs = predict(
    mirna_seqs=["UGAGGUAGUAGGUUGUAUAGUU"],
    target_seqs=["ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU"],
)

print(f"Interaction probability: {probs[0]:.4f}")
print(f"Prediction: {'INTERACTION' if probs[0] >= 0.5 else 'NO INTERACTION'}")
