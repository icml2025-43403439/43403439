import os
import random
import numpy as np
import math
import ast
from typing import Optional
import requests
from pathlib import Path
import zipfile
import shutil

def get_dataset(data: str, n: Optional[int] = None, folder = "./", info_str: bool = False):
    """
    Parameters:
    ----------
    data (str): Must be either "weaving", "rsk", "schubert", "quiver", "mheight", "symmetric_group_char", "grassmannian_cluster_algebras", "kl_polynomial", or "lattice_path"
    n (int): 
        - n = 6, 7, or 8 for "weaving"
        - n = 8, 9, or 10 for "rsk"
        - n = 3, 4, 5, or 6 for "schubert"
        - n = 10, 11, or 12 for "mheight"
        - n = 18, 20, 22 for "symmetric_group_char"
        - n = 5,6,7 for "kl_polynomial"
        - n = 10, 11, 12, or 13 for "lattice_path"
        - There are not multiple values of n for the "quiver" and "grassmannian_cluster_algebras" datasetes
    folder (str, optional): Base directory for dataset files. Defaults to "./".
    info_str (bool, optional): Also return a string with information about the dataset. Defaults to False.

    Returns:
    --------
    tuple: A tuple containing the following elements: X_train (np.array), y_train (np.array), X_test (np.array), y_test (np.array), input_size (int), output_size (int), num_tokens (int)
    info_str (str, optional): A string with information about the dataset.
    """

    if data == "weaving":
        assert n in {6, 7, 8}, f"Can't handle n={n}. n must be 6, 7, or 8."

        X_train = [
            ast.literal_eval(line)
            for line in open( os.path.join(folder, f"weaving_patterns/weaving_pattern_train_{n}.txt"), 'r')
        ]
        X_test = [
            ast.literal_eval(line)
            for line in open( os.path.join(folder, f"weaving_patterns/weaving_pattern_test_{n}.txt"), 'r')
        ]
        y_train = [ast.literal_eval(line) for line in open( os.path.join(folder, f"weaving_patterns/labels_train_{n}.txt"), 'r')
                ]
        y_test = [ast.literal_eval(line) for line in open( os.path.join(folder, f"weaving_patterns/labels_test_{n}.txt"), 'r')
                ]
        
        input_size = len(X_train[0])
        output_size = 2
        
        num_tokens = np.max(X_train) + 1
        
        
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Inputs are sequences of length {input_size} with entries between 0 and {num_tokens-1}, representing weaving patterns.")
        print(f"There are {output_size} classes. Weaving patterns are labeled 1, non-weaving patterns are labeled 0.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInputs are sequences of length {input_size} with entries between 0 and {num_tokens-1}, representing weaving patterns.\nThere are {output_size} classes. Weaving patterns are labeled 1, non-weaving patterns are labeled 0."
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens)

    elif data == "rsk":
        assert n in {8, 9, 10}, f"Can't handle n={n}. n must be 8, 9, or 10."
        base_path = os.path.join(folder, "./robinson-schensted/input_permutations")
        
        X_train = process_rsk(n, folder, "train")
        X_test = process_rsk(n, folder, "test")

        max_input_length = max( max([len(x) for x in X_train]),  max([len(x) for x in X_test]) )

        X_train_padded = [ np.array(row + [n+2]*(max_input_length - len(row) ) ) for row in X_train]
        X_test_padded  = [ np.array( row + [n+2]*(max_input_length - len(row) ) ) for row in X_test]
        
        y_train_permutation = [
                    ast.literal_eval(line)
                    for line in open(f"{base_path}_{n}_train.txt", 'r')
                ]
        
        y_train = [inversion_vector(p) for p in y_train_permutation]
        
        y_test_permutation = [
                    ast.literal_eval(line)
                    for line in open(f"{base_path}_{n}_test.txt", 'r')
                ]
        
        y_test = [inversion_vector(p) for p in y_test_permutation]
        
        output_size = len(y_train[0])
        num_tokens = n+3
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Input sequence is length {max_input_length} with entries 0 through {num_tokens-1}, representing two concatenated SSYT, padded so that all inputs have the same length.")
        print(f"Outputs are binary sequences of length {len(y_train[0])}. Output is one permutation represented by its inversion sequence.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInput sequence is length {max_input_length} with entries 0 through {num_tokens-1}, representing two concatenated SSYT, padded so that all inputs have the same length.\nOutputs are binary sequences of length {len(y_train[0])}. Output is one permutation represented by its inversion sequence."
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), max_input_length, output_size, num_tokens), info_str
        else:
            return (np.array(X_train_padded), np.array(y_train), np.array(X_test_padded), np.array(y_test), max_input_length, output_size, num_tokens)

    elif data == "schubert":
        assert n in {3, 4, 5, 6}, f"Can't handle n={n}. n must be 3, 4, 5, or 6."

        max_n = 2*n-1
        X_train = [
                    ast.literal_eval(line)[:3]
                    for line in open( os.path.join(folder, f"schubert_polynomial_coeff/schubert_structure_coefficients_triples_{n}_train.txt"), 'r')
                ]

        y_train = [
                    ast.literal_eval(line)[3:][0]
                    for line in open( os.path.join(folder, f"schubert_polynomial_coeff/schubert_structure_coefficients_triples_{n}_train.txt"), 'r')
                ]
        X_test = [
                    ast.literal_eval(line)[:3]
                        for line in open( os.path.join(folder, f"schubert_polynomial_coeff/schubert_structure_coefficients_triples_{n}_test.txt"), 'r')
                ]
        y_test = [
                    ast.literal_eval(line)[3:][0]
                        for line in open( os.path.join(folder, f"schubert_polynomial_coeff/schubert_structure_coefficients_triples_{n}_test.txt"), 'r')
                ]
        X_train_flattened = [row[0] + row[1] + row[2] + list(range( len(row[2]) +1, max_n+1)) for row in X_train]
        X_test_flattened = [row[0] + row[1] + row[2] + list(range( len(row[2]) +1 , max_n+1)) for row in X_test]

        input_size = len(X_train_flattened[0])
        output_size = max(max(y_train), max(y_test) ) + 1
        num_tokens =  max_n+1 
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Inputs are sequences of length {input_size}, which represent three concatenated permutations on the letters 1 through {num_tokens-1}.")
        print(f"There are {output_size} classes, which give the structure constant for the input permutations.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInputs are sequences of length {input_size}, which represent three concatenated permutations on the letters 1 through {num_tokens-1}.\nThere are {output_size} classes, which give the structure constant for the input permutations."
            return (np.array(X_train_flattened), np.array(y_train), np.array(X_test_flattened), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train_flattened), np.array(y_train), np.array(X_test_flattened), np.array(y_test), input_size, output_size, num_tokens)


    elif data == "symmetric_group_char":
        assert n in {18, 20, 22}, f"Can't handle n={n}. n must be 18, 20, or 22."

        train = [
                ast.literal_eval(line)
                for line in open( os.path.join(folder, f"symmetric_group_char/sym_grp_char_{n}_train.txt"), 'r')
            ]
        test = [
                ast.literal_eval(line)
                for line in open( os.path.join(folder, f"symmetric_group_char/sym_grp_char_{n}_test.txt"), 'r')
            ]
        input_size = 2*n
        X_train = [  p1 + [0]*(n- len(p1) ) + p2 + [0]*(n- len(p2) )   for (p1, p2, char) in train]
        y_train = [char for (p1, p2, char) in train]
        X_test = [ p1 + [0]*(n- len(p1) ) +   p2 + [0]*(n- len(p2) )  for (p1, p2, char) in test]
        y_test = [char for (p1, p2, char) in test]

        output_size = 1
        num_tokens = max(np.max(X_train), np.max(X_test)) + 1
        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        min_val = min(np.min(y_train), np.min(y_test))
        y_train, y_test = y_train,  y_test
        print(y_train[:10])
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Inputs are sequences of length {input_size} with entries 0 through {num_tokens-1}, which represent two concatenated integer partitions of n={n}.")
        print(f"There are {output_size} classes for n={n}.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInputs are sequences of length {input_size} with entries 0 through {num_tokens-1}, which represent two concatenated integer partitions of n={n}.\nThere are {output_size} classes for n={n}."
            return (np.array(X_train.reshape(X_train.shape[0], -1)), np.array(y_train), np.array(X_test.reshape(X_test.shape[0], -1)), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train.reshape(X_train.shape[0], -1)), np.array(y_train), np.array(X_test.reshape(X_test.shape[0], -1)), np.array(y_test), input_size, output_size, num_tokens)

    elif data == "quiver":
        path_to_files = os.path.join(folder, "./cluster_algebra_quivers/")
        train_data, test_data = load_quiver_data(path_to_files)

        X_train_unshuffled = np.array([data[0] for data in train_data])
        y_train_unshuffled = np.array([data[1] for data in train_data])
        X_test_unshuffled = np.array([data[0] for data in test_data])
        y_test_unshuffled = np.array([data[1] for data in test_data])

        X_train, y_train = shuffle_data(X_train_unshuffled, y_train_unshuffled)
        X_test, y_test = shuffle_data(X_test_unshuffled, y_test_unshuffled)

        input_size = len(X_train[0])
        output_size = len(set(y_train))  # Assuming unique classes from y_train
        num_tokens = max(len(np.unique(X_train)), len(np.unique(X_test))) + 1
        rescale = max( np.abs(np.min(X_train)),  np.abs(np.min(X_test)) )
        X_train, X_test = X_train + rescale, X_test + rescale
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Input sequences of length {input_size} are flattened adjacency matrices with entries 0 through {num_tokens-1}")
        print(f"There are {output_size} classes: A_11: 0, BD_11: 1, D_11: 2, BE_11: 3, BB_11: 4, E_11: 5, DE_11: 6")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInput sequences of length {input_size} are flattened adjacency matrices with entries 0 through {num_tokens-1}\nThere are {output_size} classes: A_11: 0, BD_11: 1, D_11: 2, BE_11: 3, BB_11: 4, E_11: 5, DE_11: 6"
            return (np.array(X_train.reshape(X_train.shape[0], -1)), np.array(y_train), np.array(X_test.reshape(X_test.shape[0], -1)), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train.reshape(X_train.shape[0], -1)), np.array(y_train), np.array(X_test.reshape(X_test.shape[0], -1)), np.array(y_test), input_size, output_size, num_tokens)

    elif data == "mheight":
        assert n in {8, 9, 10, 11, 12}, f"Can't handle n={n}. n must be 8, 9, 10, 11 or 12."

        base_path = os.path.join(folder, "./mheight_function/mHeight")

        #We filtered out all classes that contained less than 0.01% of the data
        largest_class = 4

        mheight_train = np.loadtxt(f"{base_path}_{n}_train.txt", dtype = str, delimiter = ";")
        mheight_test = np.loadtxt(f"{base_path}_{n}_test.txt", dtype = str, delimiter = ";")

        X_train, y_train = parse_mheight_data(mheight_train)
        X_test, y_test = parse_mheight_data(mheight_test)

        num_classes = len(set(y_train+y_test))
        
        input_size = len(X_train[0])
        output_size = num_classes
        num_tokens = n
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Input sequences are permutations represented by their inversion sequence, which is a binary sequence of length ({n} choose 2)= {input_size}.")
        print(f"There are {output_size} classes; classes that contained less than 0.01% of the data were filtered.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInput sequences are permutations represented by their inversion sequence, which is a binary sequence of length ({n} choose 2)= {input_size}.\nThere are {output_size} classes; classes that contained less than 0.01% of the data were filtered."
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens)


    elif data == "grassmannian_cluster_algebras":
        base_path = os.path.join(folder, "grassmannian_cluster_algebras/3_4_12")
        X_train = [
            ast.literal_eval(str(line).replace("][", "],["))
            for line in open(f'{base_path}_valid_train.txt', 'r')
        ] + [
            ast.literal_eval(str(line).replace("][", "],["))
            for line in open(f'{base_path}_invalid_train.txt', 'r')
        ]
        X_test = [
            ast.literal_eval(str(line).replace("][", "],["))
            for line in open(f'{base_path}_valid_test.txt', 'r')
        ] + [
            ast.literal_eval(str(line).replace("][", "],["))
            for line in open(f'{base_path}_invalid_test.txt', 'r')
        ]
        y_train = [1] * (len(X_train) // 2 )+ [0] * (len(X_train) // 2)
        y_test = [1] * (len(X_test) // 2) + [0] * (len(X_test) // 2)
        input_size = len(X_train[0]) * len(X_train[0][0])  # Assuming all data points have the same shape
        output_size = 2  # Valid or invalid
        X_train_unshuffled, y_train_unshuffled, X_test_unshuffled, y_test_unshuffled = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        
#
        X_train, y_train = shuffle_data(X_train_unshuffled, y_train_unshuffled)
        X_test, y_test = shuffle_data(X_test_unshuffled, y_test_unshuffled)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        num_tokens = max(np.max(X_train), np.max(X_test)) + 1
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Inputs are sequences of length {input_size}, with {num_tokens} tokens, which represent 3x4 SSYT")
        print(f"There are {output_size} classes. SSYT that index a valid cluster variable are labeled 1 and SSYT that do not are labeled 0.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInputs are sequences of length {input_size}, with {num_tokens} tokens, which represent 3x4 SSYT\nThere are {output_size} classes. SSYT that index a valid cluster variable are labeled 1 and SSYT that do not are labeled 0."
            return (np.array(X_train.reshape(X_train.shape[0], -1)), np.array(y_train), np.array(X_test.reshape(X_test.shape[0], -1)), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train.reshape(X_train.shape[0], -1)), np.array(y_train), np.array(X_test.reshape(X_test.shape[0], -1)), np.array(y_test), input_size, output_size, num_tokens)

    elif data == "kl_polynomial":
        assert n in {4, 5, 6, 7, 8}, f"Can't handle n={n}. n must be 8, 9, or 10."

        path_to_files = os.path.join(folder, "kl-polynomials/")
        train_data, test_data = load_kl_polynomial_data(path_to_files, n)

        # Extracting features and labels from the loaded data
        # Assuming each datum contains three lists: two for features and one for labels
        X_train = np.array([np.concatenate((datum[0], datum[1])) for datum in train_data])
        max_coeff_train = max([len(i[2]) for i in train_data])
        max_coeff_test = max([len(i[2]) for i in test_data])
        max_coeff = max(max_coeff_train,max_coeff_test)

        # Pad polynomials with zero coefficients
        for i in train_data:
            temp = i[2]
            temp = temp + (max_coeff - len(temp))*[0]
            i[2] = temp
        for i in test_data:
            temp = i[2]
            temp = temp + (max_coeff - len(temp))*[0]
            i[2] = temp
        
        y_train = np.array([datum[2] for datum in train_data])
        X_test = np.array([np.concatenate((datum[0], datum[1])) for datum in test_data])
        y_test = np.array([datum[2] for datum in test_data])

        input_size = len(X_train[0])  # Assuming all feature vectors are of the same size
        output_size = max(np.max(y_train), np.max(y_test)) + 1
        num_tokens = max(np.max(X_train), np.max(X_test)) + 1
        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Inputs are sequences of length {input_size}, representing two permutations on the letters 0 through {num_tokens-1}")
        print(f"There are {output_size} classes, which each represent the fifth coefficient in the polynomial.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInputs are sequences of length {input_size}, representing two permutations on the letters 0 through {num_tokens-1}\nThere are {output_size} classes, which each represent the fifth coefficient in the polynomial."
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens)

    elif data == "lattice_path":
        assert n in {10, 11, 12, 13}, f"Can't handle {n}"
        file_path = os.path.join(folder, "./lattice_paths/")

        # Determine the specific file names based on the given 'n'
        size = f"{n}_{n-1}"

        # Load train and test data for the specified size
        train_data, test_data = load_lattice_path_dataset(size, file_path)

        # Extract features and labels from the loaded data
        X_train_unshuffled = np.array([data[0] for data in train_data])
        y_train_unshuffled = np.array([data[1] for data in train_data])
        X_test_unshuffled = np.array([data[0] for data in test_data])
        y_test_unshuffled = np.array([data[1] for data in test_data])

        X_train, y_train = shuffle_data(X_train_unshuffled, y_train_unshuffled)
        X_test, y_test = shuffle_data(X_test_unshuffled, y_test_unshuffled)
        
        input_size = len(X_train[0])
        output_size = 2
        num_tokens = max(np.max(X_train), np.max(X_test)) + 1

        print(f"Train set has {len(X_train)} examples")
        print(f"Test set has {len(X_test)} examples")
        print(f"Inputs are two concatenated binary sequences represented a lattice path and its cover. The input for n={n} is length {input_size}.")
        print(f"There are {output_size} classes. Lagrange covers are labeled 0, matching covers are labeled 1.")
        if info_str:
            info_str = f"Train set has {len(X_train)} examples\nTest set has {len(X_test)} examples\nInputs are two concatenated binary sequences represented a lattice path and its cover. The input for n={n} is length {input_size}.\nThere are {output_size} classes. Lagrange covers are labeled 0, matching covers are labeled 1."
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens), info_str
        else:
            return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), input_size, output_size, num_tokens)

    else:
        raise NotImplementedError(f'No {data}. Supported options are "weaving", "rsk", "schubert", "quiver", "mheight", "symmetric_group_char", "grassmannian_cluster_algebras", "kl_polynomial", or "lattice_path".')



def parse_mheight_data(mheight_set):
    #sequences = [ [int(x) for _, x in enumerate(row.split(';')[0])] for _, row in enumerate(mheight_set) if int(row.split(';')[1]) <= largest_class ]
    #labels = [ int(row.split(';')[1]) for _, row in enumerate(mheight_set) if int(row.split(';')[1]) <= largest_class ]
    sequences = [i[0].replace('(', '').replace(')', '') for i in mheight_set]
    sequences = [i.split(",") for i in sequences]
    sequences = [[int(j) for j in i] for i in sequences]
    labels = [int(i[1]) for i in mheight_set]
    return sequences, labels


def load_lattice_path_dataset(size, file_path):
    '''Helper function for loading the lattice path data'''
    orders = ['lagrange', 'matching']
    split = ['train', 'test']
    poset_label = {'lagrange': 0, 'matching': 1}
    train_data = []
    test_data = []
    sizes = size.split('_')
    size1 = int(sizes[0])
    size2 = int(sizes[1])

    for order in orders:
        for mode in split:
            file = open(file_path + order + '_covers_' + mode + '_' + size + '.csv')
            while True:
                content = file.readline()
                if not content:
                    break
                # Process content to remove unwanted characters and split correctly
                content = content.strip().replace('\'', '').replace(' ', '').replace(',', '')
                # Convert all elements to integers
                content = [int(char) for char in content if char.isdigit()]

                data_entry = [content, poset_label[order]]
                if mode == 'train':
                    train_data.append(data_entry)
                else:
                    test_data.append(data_entry)
            file.close()

    
    return train_data, test_data


def load_quiver_data(path_to_files):
    '''Helper function for loading the quiver data.'''
    # Names of data files
    file_names = [
        'A_11_bmatrices_test.csv',
        'BD_11_depth9_bmatrices_train.csv',
        'D_11_bmatrices_test.csv',
        'A_11_bmatrices_train.csv',
        'BE_11_depth8_bmatrices_test.csv',
        'D_11_bmatrices_train.csv',
        'BB_11_depth10_bmatrices_test.csv',
        'BE_11_depth8_bmatrices_train.csv',
        'E_11_depth9_bmatrices_test.csv',
        'BB_11_depth10_bmatrices_train.csv',
        'DE_11_depth9_bmatrices_test.csv',
        'E_11_depth9_bmatrices_train.csv',
        'BD_11_depth9_bmatrices_test.csv',
        'DE_11_depth9_bmatrices_train.csv'
    ]

    # Class symbols
    class_names = {
        'A_11': 0,
        'BD_11': 1,
        'D_11': 2,
        'BE_11': 3,
        'BB_11': 4,
        'E_11': 5,
        'DE_11': 6
    }

    train_data = []
    test_data = []

    # Load data from files
    for f in file_names:
        file = open(path_to_files + f, "r")
        class_name = f.split('_')
        name = class_name[0] + '_' + class_name[1]

        while True:
            content = file.readline()
            if not content:
                break
            content = content.split(',')
            content = [int(i) for i in content if (i.isdigit() or i[0] == '-')]

            if 'train' in f:
                train_data.append([content, class_names[name]])
            elif 'test' in f:
                test_data.append([content, class_names[name]])

        file.close()
    return train_data, test_data


def load_kl_polynomial_data(path_to_files,size):

    # Names of data files

    file_names = {4:['kl_polynomials_4_train.txt','kl_polynomials_4_test.txt'],
                  5:['kl_polynomials_5_train.txt','kl_polynomials_5_test.txt'],
                  6:['kl_polynomials_6_train.txt','kl_polynomials_6_test.txt'],
                  7:['kl_polynomials_7_train.txt','kl_polynomials_7_test.txt']}

    # Lists to store train and test as tuples
    train_data = []
    test_data = []

    # Load valid train Young diagrams

  #  print(file_names[size])

    for k,t in enumerate(file_names[size]):

        file = open(path_to_files+t, "r")
        while True:
            content=file.readline()
            if not content:
                break
            content = content.split(",")
            perm1 = list(content[0])
            perm2 = list(content[1])[1:]
            coeffs = content[2:]
            coeffs[-1] = coeffs[-1][:-1]
            perm1 = [int(i) for i in perm1]
            perm2 = [int(i) for i in perm2]
            coeffs = [int(i) for i in coeffs]
            datum = [perm1,perm2,coeffs]
            if k == 0:
                train_data.append(datum)
            else:
                test_data.append(datum)
        file.close()
    return train_data, test_data

def process_rsk(n, folder, train_or_test = "train"):
    base_path = os.path.join(folder, "./robinson-schensted/output_tableau_pairs")
    processed = []
    with open(f"{base_path}_{n}_{train_or_test}.txt", 'r') as f:
        for line in f:
            # Replace '[' with '0' and ']' with '9'
            modified_line = line.replace('[', '0,').replace(']', f',{n+1}')
            # Convert the string representation to an actual list
            processed_list = ast.literal_eval(modified_line)
            processed.append(list(processed_list))
    return processed



def inversion_vector(permutation):
    """
    Converts permutation to inversion vector format
    """
    ret = []
    n = len(permutation)
    for i in range(n):
      for j in range(i+1,n):
        if permutation[i] > permutation[j]:
          ret.append(1)
        else:
          ret.append(0)
    return ret

def shuffle_data(sequences, labels, s = 32):
    random.seed(s)
    data = list(zip(sequences, labels))
    random.shuffle(data)
    sequences, labels = zip(*data)
    return sequences, labels

# A helper function to handle the Google Drive confirmation mechanism for large files.
def download_file_from_google_drive(file_id, destination):
    """
    Downloads a file from Google Drive using its file_id and saves to 'destination'.
    """
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    _save_response_content(response, destination)

def _get_confirm_token(response):
    """
    For large files, Google may prompt for confirmation. This function fetches 
    the confirmation token if present.
    """
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def _save_response_content(response, destination):
    """
    Writes the streamed response content to destination.
    """
    chunk_size = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)

def main():
    """
    Main function to orchestrate the downloads for each dataset, then unzip
    each archive into its corresponding data/ subfolder.
    """

    datasets_to_download = {
        "grassmannian_cluster_algebras": {
            "id": "1Dd4PAOgm7bAtXSGmQW81OE-O_7dS7qU_", 
            "filename": "grassmannian_data.zip"
        },
        "kazhdan_lusztig_polynomial_coefficients": {
            "id": "1A9swYSBVM4Y5KAFC52AzVRMVshVz4yyR", 
            "filename": "kl_polynomials.zip"
        },
        "lattice_path_posets": {
            "id": "1Wm9mtZQjXXQ4rl0TU9KtJ1T4RQaGsJNz", 
            "filename": "lattice_path_posets_data.zip"
        },
        "quiver_mutation_equivalence": {
            "id": "1UmRLOhNq2mX6s4NQPIgciuGG9HfvrKWC", 
            "filename": "quiver_mutation_data.zip"
        },
        "schubert_polynomial_structure_constants": {
            "id": "15bERRWWue-3gKSir3hVhfejNTeZJgsl9", 
            "filename": "schubert_polynomial_coeff.zip"
        },
        "weaving_patterns": {
            "id": "1HsWuHpTkCOtpyTG2dFH49jzkKIZYwKG8", 
            "filename": "weaving_patterns_data.zip"
        },
        "mheight_function": {
            "id": "1NteiP494xpQ4KzR9dVUaDhNtUPnumeuX", 
            "filename": "mheight_function_data.zip"
        },
        "rsk": {
            "id": "1CfuxD_XgTefbEduxJnXgXoUOt-GY-smq", 
            "filename": "rsk.zip"
        },
        "symmetric_group_characters": {
            "id": "15AHAn9NnC7crzG_8BnaH3pp1aOGUUniV", 
            "filename": "symmetric_group_data.zip"
        }
    }

    # Create an output directory for storing .zip files before unzipping
    output_directory = Path("datasets")
    output_directory.mkdir(exist_ok=True)

    for dataset_name, info in datasets_to_download.items():
        file_id = info["id"]
        filename = info["filename"]
        zip_destination = output_directory / filename

        # Download phase
        print(f"Downloading {dataset_name} from Google Drive (file_id={file_id}) to {zip_destination}...")
        download_file_from_google_drive(file_id, zip_destination)
        print(f"Finished downloading {dataset_name}.")

        # Unzip phase: place each dataset into data/dataset_name/
        data_subfolder = Path("data")
        if dataset_name == "weaving_patterns":
            data_subfolder = data_subfolder / "weaving_patterns"
        data_subfolder.mkdir(parents=True, exist_ok=True)
        print(f"Unzipping {filename} into {data_subfolder}...")

        try:
            with zipfile.ZipFile(zip_destination, 'r') as zip_ref:
                zip_ref.extractall(data_subfolder)
            print(f"Extraction complete for {dataset_name}.")
            
            # Remove __MACOSX directories
            macosx_dirs = list(Path(data_subfolder).glob('**/__MACOSX'))
            for macosx_dir in macosx_dirs:
                print(f"Removing macOS metadata directory: {macosx_dir}")
                shutil.rmtree(macosx_dir)
            
            print(f"Cleanup complete for {dataset_name}.\n")
        except zipfile.BadZipFile:
            print(f"[ERROR] {filename} is not a valid zip file - you will need to download the data manually.\n")

if __name__ == "__main__":
    main()
