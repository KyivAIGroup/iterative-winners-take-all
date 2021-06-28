import numpy as np


def check_randomness(matrix: np.ndarray):
    from ApproximateEntropy import ApproximateEntropy
    from Complexity import ComplexityTest
    from CumulativeSum import CumulativeSums
    from FrequencyTest import FrequencyTest
    from Matrix import Matrix
    from RandomExcursions import RandomExcursions
    from RunTest import RunTest
    from Serial import Serial
    from Spectral import SpectralTest
    from TemplateMatching import TemplateMatching
    from Universal import Universal

    matrix = matrix.ravel()

    if np.unique(matrix).tolist() == [0, 1]:
        # already binary matrix
        matrix_bits = matrix
    elif matrix.dtype == np.uint8:
        matrix_bits = np.unpackbits(matrix)
    else:
        matrix_bytes = matrix.tobytes()
        matrix_uint8 = np.frombuffer(matrix_bytes, dtype=np.uint8)
        matrix_bits = np.unpackbits(matrix_uint8)
    matrix_bits = ''.join(map(str, matrix_bits))

    print('The statistical test of the Binary Expansion of e')
    print('2.01. Frequency Test:\t\t\t\t\t\t\t\t', FrequencyTest.monobit_test(matrix_bits))
    print('2.02. Block Frequency Test:\t\t\t\t\t\t\t', FrequencyTest.block_frequency(matrix_bits))
    print('2.03. Run Test:\t\t\t\t\t\t\t\t\t\t', RunTest.run_test(matrix_bits))
    print('2.04. Run Test (Longest Run of Ones): \t\t\t\t', RunTest.longest_one_block_test(matrix_bits))
    print('2.05. Binary Matrix Rank Test:\t\t\t\t\t\t', Matrix.binary_matrix_rank_text(matrix_bits))
    print('2.06. Discrete Fourier Transform (Spectral) Test:\t', SpectralTest.sepctral_test(matrix_bits))
    print('2.07. Non-overlapping Template Matching Test:\t\t', TemplateMatching.non_overlapping_test(matrix_bits, '000000001'))
    print('2.08. Overlappong Template Matching Test: \t\t\t', TemplateMatching.overlapping_patterns(matrix_bits))
    print('2.09. Universal Statistical Test:\t\t\t\t\t', Universal.statistical_test(matrix_bits))
    print('2.10. Linear Complexity Test:\t\t\t\t\t\t', ComplexityTest.linear_complexity_test(matrix_bits))
    print('2.11. Serial Test:\t\t\t\t\t\t\t\t\t', Serial.serial_test(matrix_bits))
    print('2.12. Approximate Entropy Test:\t\t\t\t\t\t', ApproximateEntropy.approximate_entropy_test(matrix_bits))
    print('2.13. Cumulative Sums (Forward):\t\t\t\t\t', CumulativeSums.cumulative_sums_test(matrix_bits, 0))
    print('2.13. Cumulative Sums (Backward):\t\t\t\t\t', CumulativeSums.cumulative_sums_test(matrix_bits, 1))
    result = RandomExcursions.random_excursions_test(matrix_bits)
    print('2.14. Random Excursion Test:')
    print('\t\t STATE \t\t\t xObs \t\t\t\t P-Value \t\t\t Conclusion')
    
    for item in result:
        print('\t\t', repr(item[0]).rjust(4), '\t\t', item[2], '\t\t', repr(item[3]).ljust(14), '\t\t',
              (item[4] >= 0.01))
    
    result = RandomExcursions.variant_test(matrix_bits)
    
    print('2.15. Random Excursion Variant Test:\t\t\t\t\t\t')
    print('\t\t STATE \t\t COUNTS \t\t\t P-Value \t\t Conclusion')
    for item in result:
        print('\t\t', repr(item[0]).rjust(4), '\t\t', item[2], '\t\t', repr(item[3]).ljust(14), '\t\t',
              (item[4] >= 0.01))
