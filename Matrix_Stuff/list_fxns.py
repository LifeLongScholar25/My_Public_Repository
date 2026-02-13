"""
Author: Drake Galvan Smith
DOC: 12.22.2025
Last Modified: 1.13.2026
Purpose: To create some more list methods that will work well with the
    NumMatrix class.
"""

#list_bool_fxns.py

def row_of_zeros(in_list:list):
    """Returns True if the row list is full of zeros, and False if
        the row list has anything but pure zeros."""
    for entry in in_list:
        if entry != 0:
            return False
    return True #Only is reached if all entries are 0's

def first_non_zero_index(in_list:list):
    """Returns the index value of the first non-zero entry in a row list
        that is not a row of zeroes itself. Python indexing."""
    if not row_of_zeros(in_list):
        pos_counter = 0
        for entry in in_list:
            if entry == 0:
                pos_counter += 1
                #Number preemptively incremented.
            else:
                return pos_counter
    else: #I.e. a row of zeros.
        raise RuntimeError ("This function cannot be used with zero rows.")

def same_dim_rv(test_list:list):
    """Returns True if a collection of lists collectively 
        have the same length. If they do not all match,  
        returns False."""
    sub_llen = len(test_list[0]) #Accesses first sublist and gets length.
    #Even if first list could be only outstanding one, if even
    #one is out of align, returns False.
    for sub_list in test_list:
        if len(sub_list) != sub_llen:
            return False
    return True

def num_zeroes(in_list:list):
    """Receives a list, returns the number of 0's present."""
    counter = 0
    for val in in_list:
        if isinstance(val,(int,float)):
            if val == 0:
                counter += 1
        else:
            raise TypeError ("Only integer or floats allowed.")
    return counter

def num_non_ones(in_list:list):
    """Receives a list, returns the number of non-ones present."""
    counter = 0
    for val in in_list:
        if isinstance(val,(int,float)):
            if val != 1:
                counter += 1
        else:
            raise TypeError ("Only integers or floats allowed.")
    return counter

def ret_num_int_float(in_list:list):
    """Returns the total number of integers and floats in a list."""
    int_or_float = 0
    for item in in_list:
        if isinstance(item,(int,float)):
            int_or_float += 1
    return int_or_float
