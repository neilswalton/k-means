#########################################
#                                       #
# Useful utility functions for the      #
# k-means project.                      #
#                                       #
# Author: Neil Walton                   #
# Date: April 2018                      #
#                                       #
#########################################

import csv
import numpy as np

def csv_to_arrays(filename):
    """
    Given a csv file, return a list
    of numpy array, each representing
    a data point
    """

    with open(filename, 'rt') as file:
        
        reader = csv.reader(file)
        _ = reader.__next__() #ignore the column headers
        data = []

        for row in reader:
            data.append(np.array(row).astype(float))

    return data