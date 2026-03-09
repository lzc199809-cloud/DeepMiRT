#!/usr/bin/env python3
"""
Data Preprocessing Utilities — RNA Sequence Format Conversion Module

This module converts DNA-notation sequences in the dataset to the RNA notation
format required by the RNA-FM model.

[Why is this conversion needed?]
- The RNA-FM model was trained on RNA sequences and expects input in RNA notation: A, U, G, C
- Our dataset stores sequences in DNA notation: A, T, G, C (where T replaces U)
- During training, DNA notation T must be converted to RNA notation U to match the model's expected input format

[Architecture Position]
- This module is called by Dataset.__getitem__() during training
- The conversion happens at the data loading stage without modifying the original CSV files
- Reference: finalize_dataset.py:86-93 performs the reverse operation (U→T) for data export

[Design Decisions]
- Conversion is performed online (in the Dataset) rather than preprocessing the CSV, to preserve original data integrity
- All sequences are converted to uppercase to ensure format consistency
- The character N (representing ambiguous bases) is allowed; RNA-FM can handle ambiguous bases
"""

from __future__ import annotations


def dna_to_rna(seq: str) -> str:
    """
    Convert a DNA-notation sequence to an RNA-notation sequence.

    [Description]
    - Converts T (thymine, DNA) to U (uridine, RNA)
    - Converts to uppercase
    - Removes all whitespace characters
    - Idempotent: sequences already in RNA format remain unchanged

    [Design Decisions]
    - Why convert online? To keep the original CSV data intact for auditing and reproducibility
    - Why uppercase? To ensure consistency with the RNA-FM model's expected input format
    - Why allow N? RNA-FM's tokenizer can handle ambiguous bases

    Args:
        seq (str): DNA-notation sequence string, may contain A, T, G, C, N and whitespace

    Returns:
        str: RNA-notation sequence string, containing A, U, G, C, N (uppercase, no whitespace)

    Example:
        >>> dna_to_rna('ATCGATCG')
        'AUCGAUCG'
        >>> dna_to_rna('atcg')  # mixed case
        'AUCG'
        >>> dna_to_rna('AUCGAUCG')  # already RNA format (idempotent)
        'AUCGAUCG'
        >>> dna_to_rna('ATC NGATCG')  # contains N and whitespace
        'AUCNGAUCG'
        >>> dna_to_rna(' ATC G ')  # leading/trailing whitespace
        'AUCG'
    """
    # Step 1: Convert to uppercase
    seq = str(seq).upper()
    
    # Step 2: Remove all whitespace characters (spaces, tabs, newlines)
    seq = seq.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    
    # Step 3: Convert T (DNA) to U (RNA)
    seq = seq.replace("T", "U")
    
    return seq


def validate_rna_sequence(seq: str, min_len: int = 5, max_len: int = 100) -> bool:
    """
    Validate whether a sequence is in valid RNA format.

    [Description]
    - Checks that the sequence contains only valid RNA characters: A, U, G, C, N
    - Checks that the sequence length is within the specified range
    - If it contains T, the DNA-to-RNA conversion was not performed; returns False

    [Design Decisions]
    - Why check for T? It serves as an indicator of conversion failure, aiding data flow debugging
    - Why allow N? RNA-FM's tokenizer supports ambiguous bases
    - Why impose length limits? To prevent abnormally long sequences from causing memory overflow

    Args:
        seq (str): the sequence string to validate
        min_len (int): minimum length (inclusive), default 5
        max_len (int): maximum length (inclusive), default 100

    Returns:
        bool: True if the sequence is valid, False otherwise

    Example:
        >>> validate_rna_sequence('AUCGAUCG', 5, 30)
        True
        >>> validate_rna_sequence('ATCG', 5, 30)  # contains T (DNA notation)
        False
        >>> validate_rna_sequence('AU', 5, 30)  # too short
        False
        >>> validate_rna_sequence('A' * 31, 5, 30)  # too long
        False
        >>> validate_rna_sequence('AUCNGAUCG', 5, 30)  # contains N (valid)
        True
    """
    # Check length
    if len(seq) < min_len or len(seq) > max_len:
        return False
    
    # Define valid RNA character set
    valid_chars = {"A", "U", "G", "C", "N"}
    
    # Check if all characters are valid
    for char in seq:
        if char not in valid_chars:
            # Specifically check for T, indicating conversion failure
            if char == "T":
                return False
            # Other invalid characters also return False
            return False
    
    return True


def prepare_rnafm_input(mirna_seq: str, target_seq: str) -> tuple[str, str]:
    """
    Prepare an input sequence pair for the RNA-FM model.

    [Description]
    - Converts both miRNA and target sequences to RNA notation
    - Returns two separate strings (not concatenated)
    - RNA-FM uses a shared encoder architecture that processes each sequence independently

    [Design Decisions]
    - Why not concatenate? The dual-encoder processes each sequence in separate forward passes
    - Concatenation would break the model's architectural design and degrade performance
    - Returning a tuple is convenient for use in Dataset.__getitem__()

    Args:
        mirna_seq (str): miRNA sequence (DNA notation)
        target_seq (str): target sequence (DNA notation)

    Returns:
        tuple[str, str]: (mirna_rna, target_rna) tuple, both in RNA notation

    Example:
        >>> mirna_rna, target_rna = prepare_rnafm_input('ATCG', 'TAGC')
        >>> mirna_rna
        'AUCG'
        >>> target_rna
        'UAGC'
    """
    # Convert the two sequences separately
    mirna_rna = dna_to_rna(mirna_seq)
    target_rna = dna_to_rna(target_seq)
    
    return mirna_rna, target_rna


def compute_sequence_stats(csv_path: str, sample_n: int = 10000) -> dict:
    """
    Compute statistics for sequences in a CSV file.

    [Description]
    - Samples a specified number of rows from the CSV file
    - Computes sequence length distributions, character frequencies, DNA notation detection, etc.
    - Used for data quality checks and analysis

    [Design Decisions]
    - Why lazy-import pandas? To avoid introducing a heavy dependency at module load time
    - Import only when needed, reducing startup time
    - Sampling instead of full processing speeds up statistics computation

    Args:
        csv_path (str): path to the CSV file
        sample_n (int): number of rows to sample, default 10000. If the file has fewer rows, all rows are used

    Returns:
        dict: statistics dictionary containing the following keys:
            - 'total_rows': total number of rows in the file (excluding header)
            - 'sample_rows': actual number of sampled rows
            - 'mirna_length_min': minimum miRNA length
            - 'mirna_length_max': maximum miRNA length
            - 'mirna_length_mean': mean miRNA length
            - 'target_length_min': minimum target sequence length
            - 'target_length_max': maximum target sequence length
            - 'target_length_mean': mean target sequence length
            - 'mirna_char_freq': miRNA character frequency dictionary
            - 'target_char_freq': target sequence character frequency dictionary
            - 'mirna_with_t_count': number of miRNA sequences containing T
            - 'target_with_t_count': number of target sequences containing T

    Example:
        >>> stats = compute_sequence_stats('insect_mirna_target/data/training/train.csv', sample_n=100)
        >>> print(f"Total rows: {stats['total_rows']}")
        >>> print(f"miRNA length range: {stats['mirna_length_min']}-{stats['mirna_length_max']}")
    """
    # Lazy-import pandas to avoid introducing a heavy dependency at module load time
    import pandas as pd
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Compute total number of rows
    total_rows = len(df)
    
    # Determine sample size (capped at total number of rows)
    actual_sample_n = min(sample_n, total_rows)
    
    # Sample data
    if actual_sample_n < total_rows:
        sample_df = df.sample(n=actual_sample_n, random_state=42)
    else:
        sample_df = df
    
    # Initialize statistics dictionary
    stats = {
        'total_rows': total_rows,
        'sample_rows': len(sample_df),
    }
    
    # Compute miRNA sequence statistics
    mirna_lengths = sample_df['mirna_seq'].str.len()
    stats['mirna_length_min'] = int(mirna_lengths.min())
    stats['mirna_length_max'] = int(mirna_lengths.max())
    stats['mirna_length_mean'] = float(mirna_lengths.mean())
    
    # Compute target sequence statistics
    target_lengths = sample_df['target_fragment_40nt'].str.len()
    stats['target_length_min'] = int(target_lengths.min())
    stats['target_length_max'] = int(target_lengths.max())
    stats['target_length_mean'] = float(target_lengths.mean())
    
    # Compute character frequencies
    def compute_char_freq(seq_series):
        """Compute the frequency of each character in the sequences"""
        freq = {}
        for seq in seq_series:
            seq = str(seq).upper()
            for char in seq:
                freq[char] = freq.get(char, 0) + 1
        return freq
    
    stats['mirna_char_freq'] = compute_char_freq(sample_df['mirna_seq'])
    stats['target_char_freq'] = compute_char_freq(sample_df['target_fragment_40nt'])
    
    # Count sequences containing T (DNA notation)
    stats['mirna_with_t_count'] = (sample_df['mirna_seq'].str.contains('T', case=False, na=False)).sum()
    stats['target_with_t_count'] = (sample_df['target_fragment_40nt'].str.contains('T', case=False, na=False)).sum()
    
    return stats
