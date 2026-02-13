"""
Author: Drake Galvan Smith
DOC: 1.10.2026
Last Modifed: 1.22.2026
Purpose: To give a more computationally effective method for
    assessing the value of a matrix's determinant.
"""

#determ_algorithm.py

from nummatrix import NumMatrix
from elem_matrix_fxns import *

def row_op_determinant(in_matrix):
    """Algorithm that finds determinant through row operation reduction. 
    Returns the determinant, and if it is not 0, will return the processed
    in_matrix, and the product of inverse matrices. If it is 0, it will return
    0, and None and None."""
    if isinstance(in_matrix,NumMatrix):
        if in_matrix.is_square_matrix():
            determ_prod = 0
            DIM = in_matrix._col_dim
            inverse_prod_mat = None 
            #Is current inverse product matrix that could be passed out of
            #the row_op_determinant function. Declared here for scope.
            z_row_counter = in_matrix.num_zero_rows()
            z_col_counter = in_matrix.num_zero_cols()
            if z_row_counter == 0 and z_col_counter == 0:
                col_indices = [x for x in range(1,in_matrix._col_dim+1)]
                determ_prod = 1 #Will not automatically be 0 from here on.
                for col_index in col_indices:
                    #Performs work, one column at a time.
                    while in_matrix.entries_below_rzeroes(col_index,col_index) == False:
                        #If there are still 0's below a column's diagonal position
                        #Then operations still need to be performed.
                        if in_matrix.mat_entry_at(col_index,col_index) != 0: #Diag entry is not 0.
                            #Work can begin.
                            if in_matrix.mat_entry_at(col_index,col_index) == 1:
                                tag = "a"
                                i_ind,j_ind = in_matrix.next_nonzero_down_col(col_index,col_index) #Always works down from diag.
                                constant = in_matrix.mat_entry_at(i_ind,j_ind)
                                #i_ind is also row number here. 
                                command = [i_ind,col_index,constant] #Targeting the next row (i_ind) by 
                                #subtracting k multiple of row == col_index
                                #COMMAND: E_RA: [r_i,r_j,k]
                                forward_mat,inv_mat = elem_inv_pairs(tag,DIM,command)
                                determ_prod *= det_ra(forward_mat)
                                if inverse_prod_mat == None:
                                    inverse_prod_mat = inv_mat
                                else:
                                    inverse_prod_mat = inv_mat.matrix_product(inverse_prod_mat)
                                in_matrix = inv_mat.matrix_product(in_matrix)
                            else:
                                #Row scaling needed
                                tag = "c"
                                constant = in_matrix.mat_entry_at(col_index,col_index)
                                command = [col_index,constant] #E_RC: [r_i,k]
                                forward_mat,inv_mat = elem_inv_pairs(tag,DIM,command)
                                determ_prod *= det_rc(forward_mat)
                                if inverse_prod_mat == None:
                                    #First inv mat incorporated into the inv mat prod.
                                    inverse_prod_mat = inv_mat
                                else:
                                    inverse_prod_mat = inv_mat.matrix_product(inverse_prod_mat)
                                in_matrix = inv_mat.matrix_product(in_matrix)
                        elif in_matrix.mat_entry_at(col_index,col_index) == 0 and \
                            in_matrix.entries_below_rzeroes(col_index,col_index):
                            #This occurs if the system above can be put into row echelon already. This results 
                            #in a determinant for a Diagonal matrix with a zero on the diagonal. Though not common
                            #It still would result in a zero. This happens thanks to the left and right sides reducing
                            #To echelon form independently of this column.
                            determ_prod = 0
                            return (determ_prod,None,None)
                        else:
                            #Diagonal entry was 0, need to find non-zero below.
                            next_i,next_j = in_matrix.next_nonzero_down_col(col_index,col_index)
                            #Producing a row swap.
                            tag = "s"
                            #Swap row = col_index with row = next_i
                            command = [col_index,next_i] #E_RS: [r_i,r_j]
                            forward_mat,inv_mat = elem_inv_pairs(tag,DIM,command)
                            determ_prod *= det_rs(forward_mat)
                            if inverse_prod_mat == None:
                                #For the case this is the first inv mat generated.
                                inverse_prod_mat = inv_mat
                            else:
                                inverse_prod_mat = inv_mat.matrix_product(inverse_prod_mat)
                            #Apply the inv mat for the next operation.
                            in_matrix = inv_mat.matrix_product(in_matrix)
                determ_upper_triangle = in_matrix.simple_determinant()
                #Product along the diagonal of the upper triangular matrix.
                determ_prod *= determ_upper_triangle
                return (determ_prod,in_matrix,inverse_prod_mat)
            else:
                return (determ_prod,None,None)
        else:
            raise RuntimeError ("Only square matrices can have determinants.")
    else:
        raise TypeError ("Only NumMatrix objects are allowed.")
    
test_mat = NumMatrix()
the_list = [[1,0,4,2],[3,0,5,9],[2,7,1,10],[8,6,5,9]]
test_mat.build_from_datamat(the_list)
print(test_mat)
determ,pro_mat,inv_prod_mat = row_op_determinant(test_mat)
print("det(A) =",determ)
print(pro_mat)
print(inv_prod_mat)