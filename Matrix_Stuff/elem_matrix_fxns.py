'''
Author: Drake Galvan Smith
DOC: 1.10.2026
Last Modified: 1.13.2026
Purpose: To produce a given elementary row matrix.
'''

#elem_matrix_fxns

from nummatrix import NumMatrix
import list_fxns as lf
from extra_math_fxns import reciprocal as reciprocal

#COMPLETED

def row_constant(req_dim:int,cmd_list:list):
    """Takes the required dimension and command list. Function generates 
        a row multiple elementary matrix. Normal matrix enumeration.
        Command Form: [targ_row, constant]"""
    if len(cmd_list) == 2:
        elementary_matrix = NumMatrix()
        elementary_matrix.build_identity_mat(req_dim)
        pos,constant = cmd_list
        i,j = pos,pos
        if i <= req_dim and i >= 1: #in [1,n]
            if constant != 1 and constant != 0:
                elementary_matrix.set_entry_to(i,j,constant)
                return elementary_matrix
            else:
                raise RuntimeError ("Only use scalar multiples k s.t. k =/= 1 or 0")
        else:
            raise IndexError ("Beyond the row range of the matrix.")
    else:
        raise RuntimeError ("The command length did not match.")

def row_swap(req_dim:int,cmd_list:list):
    """Takes required dimension and command list. Function generates a 
        row swap elementary matrix. Normal matrix enumeration.
        Command Form: [targ_row1, targ_row2]"""
    if len(cmd_list) == 2:
        elementary_matrix = NumMatrix()
        elementary_matrix.build_identity_mat(req_dim)
        val1,val2 = cmd_list
        if val1 == val2:
            raise RuntimeError ("Cannot swap the same row with itself.")
        else:
            #Because for any arbitrary row swap, rows i and j (not necessarily adjacent) 
            #have entries at ii and jj, when swapping rows, they become positions ji and ij.
            if val1 <= req_dim and val1 >= 1 and val2 <= req_dim and val2 >= 1:
                elementary_matrix.set_entry_to(val1,val1,0)
                elementary_matrix.set_entry_to(val2,val1,1)
                elementary_matrix.set_entry_to(val2,val2,0)
                elementary_matrix.set_entry_to(val1,val2,1)
                return elementary_matrix
            else:
                raise IndexError ("Beyond the row range of the matrix.")
    else: 
        raise RuntimeError ("The command length did not match.")

def row_addition(req_dim:int,cmd_list:list):
    """Takes required dimension and command list. Function generates a
        row addition elementary matrix. Normal matrix enumeration.
        Command Form: [targ_row, adding_row, constant]"""
    if len(cmd_list) == 3:
        elementary_matrix = NumMatrix()
        elementary_matrix.build_identity_mat(req_dim)
        val1,val2,constant = cmd_list
        if val1 == val2:
            raise RuntimeError ("Cannot add row to itself. Equivalent of row_constant.")
        else:
            if val1 <= req_dim and val1 >= 1 and val2 <= req_dim and val2 >= 1:
                #Addition done to the i-th row by a scalar multiple of the j-th column means
                #ii still is 1, but ij now is the scalar multiple we will call k.
                elementary_matrix.set_entry_to(val1,val2,constant)
                return elementary_matrix
            else:
                raise IndentationError ("Beyond the range of the matrix.")
    else:
        raise RuntimeError ("The command length did not match.")

#now for determinants of elementary row matrices

def is_rcem(elem_mat):
    """Boolean function that returns True if the matrix is a row constant
        elementary matrix."""
    if isinstance(elem_mat,NumMatrix):
        flat_mat = elem_mat.flatten_mat()
        l_flat_mat = len(flat_mat)
        num_int_float = lf.ret_num_int_float(flat_mat)
        if elem_mat.is_square_matrix() and num_int_float == l_flat_mat:
            num_nz = l_flat_mat - lf.num_zeroes(elem_mat.flatten_mat())
            #Should have n many non-zeroes, all of which should fall on the diagonal.
            if num_nz == elem_mat._row_dim:
                no_counter = 0 #non-ones counter
                ones_counter = 0
                diag_indices = [x for x in range(1,elem_mat._row_dim+1)]
                diag_entries = [elem_mat.mat_entry_at(x,x) for x in diag_indices]
                #THIS SHOULD ONLY HAVE one non-one on the diagonal. Determinant
                #could still be right say two -1's occurred. This should be as
                #broad as possible so as to exclude that.
                for entry in diag_entries:
                    if entry != 1 and entry != 0: #i.e. some scalar k
                        no_counter += 1 #If not a one and not a k = 0
                    if entry == 1:
                        ones_counter += 1
                #Equivalent to k and 1's allowed
                if no_counter == 1 and ones_counter == (elem_mat._row_dim - 1): #1 + (n-1) = n
                    return True
                else:
                    return False #In the case there are more non-ones than the single one
            else:
                return False
        else:
            raise RuntimeError ("EMs are only ever square matrices, and they contain only ints or floats.")
    else:
        raise TypeError ("This is only compatible with NumMatrix class.")

def is_rsem(elem_mat):
    """Boolean function that returns True if the matrix is a row swap elementary
        matrix."""
    if isinstance(elem_mat,NumMatrix):
        flat_mat = elem_mat.flatten_mat()
        l_flat_mat = len(flat_mat)
        num_int_float = lf.ret_num_int_float(flat_mat)
        if elem_mat.is_square_matrix() and num_int_float == l_flat_mat:
            z_count = flat_mat.count(0)
            o_count = flat_mat.count(1)
            num_zeros = elem_mat._row_dim*(elem_mat._col_dim-1) # Because this is so for I_n
            num_ones = elem_mat._row_dim
            if z_count == num_zeros and o_count == num_ones:
                diag_ind = [x for x in range(1,elem_mat._row_dim+1)] #1,...,n
                diag_entries = [elem_mat.mat_entry_at(x,x)for x in diag_ind]
                if diag_entries.count(0) == 2 and diag_entries.count(1) == (elem_mat._row_dim - 2):
                    return True
                else:
                    return False
            else:
                return False
        else:
            raise RuntimeError ("EMs are only ever square matrices, and they contain only ints or floats.")
    else:
        raise TypeError ("This is only compatible with NumMatrix class.")

def is_raem(elem_mat):
    """Boolean function that returns True if the matrix is a row addition
        elementary matrix."""
    if isinstance(elem_mat,NumMatrix):
        flat_mat = elem_mat.flatten_mat()
        l_flat_mat = len(flat_mat)
        num_int_float = lf.ret_num_int_float(flat_mat)
        if elem_mat.is_square_matrix() and num_int_float == l_flat_mat:
            #FOR SAKE OF KEEPING FUNCTIONS SEPARATE, row addition is not allowed
            #on the diagonal of the matrix. RCEM is for that.
            diag_ind = [x for x in range(1,elem_mat._row_dim+1)]
            diag_entries = [elem_mat.mat_entry_at(x,x) for x in diag_ind]
            num_ones = diag_entries.count(1)
            if num_ones == elem_mat._row_dim: #All 1's on the diagonal.
                num_zeroes = flat_mat.count(0)
                n = elem_mat._row_dim
                allowed_num_of_zeroes = l_flat_mat - n - 1
                if num_zeroes == allowed_num_of_zeroes: #Allowed for 1 k on non-diagonal.
                    return True
                else: # n*n - n - 1 == n(n-1) - 1. This means num 0's deviated.
                    return False
            else:
                return False
        else:
            raise RuntimeError ("EMs are only ever square matrices, and they contain only ints or floats.")
    else:
        raise TypeError ("This is only compatible with NumMatrix class.")

def det_rc(elem_mat):
    """Returns the determinant of an RCEM."""
    if is_rcem(elem_mat) and isinstance(elem_mat,NumMatrix):
        #It being a NumMatrix is a given. Done for functionality.
        diag_ind = [x for x in range(1,elem_mat._row_dim+1)]
        diag_entries = [elem_mat.mat_entry_at(x,x) for x in diag_ind]
        for entry in diag_entries:
            if entry != 1: #The scalar k
                return entry
    else:
        raise RuntimeError ("Function only accepts RCEMs.")

def det_rs(elem_mat):
    """Returns the determinant of an RSEM."""
    if is_rsem(elem_mat) and isinstance(elem_mat,NumMatrix):
        return -1
    else:
        raise RuntimeError ("Function only accepts RSEMs.")

def det_ra(elem_mat):
    """Returns the determinant of an RAEM."""
    if is_raem(elem_mat) and isinstance(elem_mat,NumMatrix):
        return 1
    else:
        raise RuntimeError ("Function only accepts RAEMs.")

def elem_inv_pairs(letter:str,dimension:int,command:list):
    """Produces a pair of elementary matrices that are inverses
        of each other. Takes in appropriate EM command and the
        appropriate tag to denote it. Returns the pair of matrices.
        Tags: RC = 'c', RA = 'a', RS = 's'. NOTE: Tags are not case-sensitive.
        RC commands here should have the scalar k for this function. Give k to
        divide by k. Row multiple for RA needs to be the negative 
        of the row multiple. To subtract a row, give the positive row mult.
        Determinant EM at left, Operation EM at right."""
    letter = letter.lower()
    #path 1
    if letter == "c": #Row Constant
        scalar_k = command[1]
        recip_k = reciprocal(scalar_k)
        command1 = [command[0],scalar_k]
        command2 = [command[0],recip_k]
        matrix1 = row_constant(dimension,command1) #Determinant EM
        matrix2 = row_constant(dimension,command2) #Operating EM
        return (matrix1,matrix2)
    #path 2
    if letter == "a": #Row Addition
        command1 = command
        command2 = [command[0],command[1],-command[2]]
        matrix1 = row_addition(dimension,command1)
        matrix2 = row_addition(dimension,command2)
        return (matrix1,matrix2)
    #path 3
    if letter == "s": #Row Swap
        matrix1 = row_swap(dimension,command)
        matrix2 = row_swap(dimension,command)
        return (matrix1,matrix2)
