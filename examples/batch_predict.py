"""Batch prediction example: process multiple miRNA-target pairs from a CSV file."""

import pandas as pd

from insect_mirna_target.predict import predict_from_csv

# Create a sample CSV
sample_data = pd.DataFrame(
    {
        "mirna_seq": [
            "UGAGGUAGUAGGUUGUAUAGUU",    # let-7a
            "UAAAGUGCUUAUAGUGCAGGUAG",   # miR-20a
            "UAGCAGCACGUAAAUAUUGGCG",    # miR-16
            "UGGAAUGUAAAGAAGUAUGUAU",    # miR-1
        ],
        "target_seq": [
            "ACUGCAGCAUAUCUACUAUUUGCUACUGUAACCAUUGAUCU",  # lin-41
            "GCAGCAUUGUACAGGGCUAUCAGAAACUAUUGACACUAAAA",  # E2F1
            "GCAAUGUUUUCCACAGUGCUUACACAGAAAUAGCAACUUUA",  # BCL2
            "UCGAAUCCAUGCAAAACAGCUUGAUUUGUUAGUACACGAAU",  # HAND2
        ],
    }
)
sample_data.to_csv("sample_input.csv", index=False)

# Run batch prediction
results = predict_from_csv(
    csv_path="sample_input.csv",
    output_path="sample_output.csv",
    device="cpu",
)

print(results[["mirna_seq", "probability", "prediction"]].to_string(index=False))
