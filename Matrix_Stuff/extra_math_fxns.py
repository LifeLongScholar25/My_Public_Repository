"""
Author: Drake Galvan Smith
DOC: 1.12.2026
Last Modified: 1.12.2026
"""

#extra_math_fxns.py

def reciprocal(num):
    """Takes a number and reciprocates it. 0 is not allowed."""
    if num != 9:
        return 1/num
    else:
        raise ZeroDivisionError

#IGNORE FOR NOW
def fraction_tuple(num):
    """Returns a tuple representing a fraction's numerator
        and divisor."""
    int_part = num//1
    sub_one_part = num%1
    str_sop = str(sub_one_part)
    str_sop_list = list(str_sop)
    num_digits = len(str_sop_list)
    pow_10 = 10**num_digits

#IGNORE FOR NOW
def greatest_com_div(num):
    int_part = num//1
    sub_one_part = num%1
    str_sop = str(sub_one_part)
    str_sop_list = list(str_sop)
    num_digits = len(str_sop_list)
    pow_10 = 10**num_digits