import numpy as np
import os
import pandas as pd


def vector_to_file(fname, x):
    try:
        os.remove(fname)
    except:
        pass

    with open(fname, 'a') as f:
        for el in x:
            f.write(str(el) + '\n')


def write_row(fname, row):
    with open(fname, 'a') as f:
        for i in range(len(row)):
            if i < (len(row)-1):
                f.write(str(row[i]) + ', ')
            else:
                f.write(str(row[i]) + '\n')

