'''
Author: Drake Galvan Smith
DOC: 12.18.2025
Last Modified: 12.21.2025
Purpose: To create a numerical matrix class that is a child
    class to the NumVector Class.
'''

#nummatrix.py

from numvector import NumVector

class NumMatrix():
    """A Numerical Matrix class that offers many different
        matrix methods and operations."""
    
    def __init__(self):
        '''Inherits the NumVector class, and initializes the NumMatrix object.'''
        self._matrix = []
        self._row_dim = None
        self._row_dim = None
    
    def __add__(self,other):
        if isinstance(other,NumMatrix):
            if self.return_mat_dim() == other.return_mat_dim():
                row_N,col_N = self.return_mat_dim()
                #Row by Row
                matrix_sum = NumMatrix()
                matrix_list = []
                for row in range(0,row_N):
                    for column in range(0,col_N):
                        entry_self = self._matrix[row][column]
                        entry_other = other._matrix[row][column]
                        matrix_list.append(entry_self + entry_other)
                matrix_sum.build_matrix(row_N,col_N,matrix_list)
                return matrix_sum
            else:
                raise RuntimeError ("Matrices do not have the same dimensions.")
        else:
            raise TypeError ("Can only add matrix objects.")
    
    def __radd__(self,other):
        return self.__add__(other)
    
    def __mul__(self,other):
        if isinstance(self,NumMatrix) and isinstance(other,(int,float)):
            row_N,col_N = self.return_mat_dim()
            scaled_matrix = NumMatrix()
            matrix_list = []
            #Row by row element-wise scalar multiplication.
            for row in range(0,row_N):
                for column in range(0,col_N):
                    entry_self = self._matrix[row][column]
                    matrix_list.append(other*entry_self)
            scaled_matrix.build_matrix(row_N,col_N,matrix_list)
            return scaled_matrix
        else:
            raise TypeError ("Can only multiply scalars against matrices with this operator.")
    
    def __rmul__(self,other):
        return self.__mul__(other)
    
    def __sub__(self,other):
        if isinstance(other,NumMatrix):
            return self + (-1 * other)
        else:
            raise TypeError ("Can only subtract matrix objects.")
    
    def __repr__(self):
        """Magic Method for determining how a matrix is printed."""
        num_rows = self._row_dim
        total_string = ""
        for val in range(0,num_rows):
            if val == 0:
                total_string += "[" + str(self._matrix[val]) + "\n"
            elif val == (num_rows-1):
                total_string += " " + str(self._matrix[val]) + "]\n"
            else:
                total_string += " " + str(self._matrix[val]) + "\n" #Accesses rows between 1 and n.
        return total_string
    
    def user_build_matrix(self):
        '''Allows the user to build a matrix through individual value
            assignment.'''
        row_count = int(input("# of Rows: "))
        col_count = int(input("# of Cols: "))
        self._row_dim = row_count
        self._col_dim = col_count
        for row_elem in range(0,row_count):
            row_list = []
            self._matrix.append(row_list)
        for row_elem in self._matrix:
            for val in range(0,col_count):
                entry = float(input("Entry: "))
                row_elem.append(entry) #row_lists
    
    def build_matrix(self,row_count:int,col_count:int,build_list:list):
        '''Takes in a list and the dimensions required for the matrix. If the list
            has the appropriate number of elements to fill a matrix of those dimensions
            then the matrix is built. If not, then an error is raised.'''
        is_all_nums = True
        for value in build_list:
            if not isinstance(value,(int,float)):
                is_all_nums = False
                break
        if is_all_nums: #True
            dim_product = row_count*col_count
            list_length = len(build_list)
            prep_matrix = []
            self._row_dim = row_count
            self._col_dim = col_count
            if list_length == dim_product:
                for num in range(0,row_count): #1,...,n produces 0,...,(n-1) for (0,n)
                    if num == (row_count-1): #Last row
                        subsection = build_list[(num*col_count):] #[a,b,...,n]
                        self._matrix.append(subsection)
                    else:
                        subsection = build_list[(num*col_count):((num+1)*col_count)]
                        self._matrix.append(subsection)
            else: #If the dimensions do not match.
                raise RuntimeError ("The number of list entries did not match the " \
                "matrix dimensionality.")
        else:
            raise TypeError ("Lists used to build matrices can only contain numbers.")
    
    def return_mat_dim(self):
        """Returns an ordered pair representing the matrix dimensions."""
        num_rows = self._row_dim
        num_cols = self._col_dim
        return [num_rows,num_cols]
    
    def build_identity_mat(self,width):
        '''Identity Matrices are only square, so function takes
            in width and returns the identity matrix I_(n)'''
        num_of_terms = width**2
        temp_list = [0 for x in range(0,num_of_terms)]
        self.build_matrix(width,width,temp_list)
        identity_matrix = self._matrix
        for val in range(0,width):
            identity_matrix[val][val] = 1
        self._matrix = identity_matrix
    
    def mat_entry_at(self,i_val,j_val):
        '''Allows for normal matrix index reference to 
            matrix entries.'''
        return self._matrix[i_val-1][j_val-1]
    
    def is_square_matrix(self):
        '''Function that returns the corresponding boolean to
            whether or not the matrix is a square matrix.'''
        row_N,col_N = self.return_mat_dim()
        if row_N == col_N:
            return True
        else: #Not a square
            return False
    
    def matrix_trace(self):
        """Returns the matrix's trace."""
        if self.is_square_matrix():
            val_list = [self._matrix[x][x] for x in range(0,self._row_dim)]
            trace = sum(val_list)
            return trace
        else:
            raise RuntimeError ("The matrix was not a square matrix.")
    
    def retrieve_col(self,col_index:int):
        '''Retrieves column from a matrix, given a column index.'''
        num_rows,num_cols = self.return_mat_dim()
        column_list = []
        for val in range(0,num_rows):
            column_list.append([])
        for val in range(0,num_rows):
            #Accesses sublist, and appends there.
            column_list[val].append(self._matrix[val][col_index])
        return column_list
    
    def matrix_transpose(self):
        '''Performs a transpose on a matrix.'''
        num_rows,num_cols = self.return_mat_dim()
        new_vector = NumVector()
        temp_matrix = NumMatrix()
        for column in range(0,num_cols):
            retrieved_column = self.retrieve_col(column)
            #Vector methods operate on ._vector
            new_vector._vector = retrieved_column
            new_vector.vector_transpose() #[[a],[b],...,[n]]
            new_row = new_vector._vector[0] #Accessing from [[a,b,...,n]]
            temp_matrix._matrix.append(new_row)
            new_vector._vector = [] #Resetting the storage list
        #Transposed matrix is reassigned to the self instance
        temp_col = self._row_dim
        self._row_dim = self._col_dim
        self._col_dim = temp_col
        self._matrix = temp_matrix._matrix
    
    def build_zero_matrix(self,row_count:int,col_count:int):
        '''Builds a zero matrix of the desired dimensions.'''
        dim_product = row_count*col_count
        temp_list = [0 for x in range(0,dim_product)]
        self.build_matrix(row_count,col_count,temp_list)
    
    def is_diagonal_matrix(self):
        '''Returns True if matrix is a diagonal matrix, returns False
            otherwise.'''
        if self.is_square_matrix():
            if self._zeros_above_diag() and self._zeros_below_diag(): #(T,T)
                return True
            else: #(F,T) (T,F) (F,F)
                return False
        else: #Failed to meet the square matrix criterion.
            return False
    
    def is_diagonal_no_zeros(self):
        '''Returns true only if all diagonal entries of a square matrix 
            are non-zero.'''
        if self.is_square_matrix():
            dim_pair = self.return_mat_dim()
            width = dim_pair[0]
            for index in range(0,width):
                if self._matrix[index][index] == 0:
                    return False
            return True
        else:
            raise RuntimeError ("This function is only useable on a square matrix.")
    
    def _zeros_below_diag(self):
        '''Returns true if there are only zeroes below the diagonal, returns False otherwise.'''
        num_rows,num_cols = self.return_mat_dim()
        for i_ind in range(0,num_rows):
            for j_ind in range(0,num_cols):
                if self._matrix[i_ind][j_ind] != 0 and i_ind > j_ind:
                    return False
        return True
    
    def _zeros_above_diag(self):
        '''Returns true if there are only zeroes above the diagonal, returns True otherwise.'''
        num_rows,num_cols = self.return_mat_dim()
        for i_ind in range(0,num_rows):
            for j_ind in range(0,num_cols):
                if self._matrix[i_ind][j_ind] != 0 and i_ind < j_ind:
                    return False
        return True
    
    def is_upper_triangle_mat(self):
        '''Returns true if the matrix is an upper triangular matrix, and returns false
            for all other cases.'''
        if self.is_square_matrix():
            if self.is_diagonal_no_zeros() and self._zeros_below_diag():
                return True
            else: #Either diagonal has 0s or fails to be zeroes all below diagonal
                return False
        else:
            raise RuntimeError ("This method can only be used on square matrices.")
    
    def is_lower_triangle_mat(self):
        '''Returns true if the matrix is a lower triangular matrix, and returns false
            for all other cases.'''
        if self.is_square_matrix():
            if self.is_diagonal_no_zeros() and self._zeros_above_diag():
                return True
            else: #Either diagonal has 0s or fails to be all zeroes above diagonal
                return False
        else:
            raise RuntimeError ("This method can only be used on square matrices.")
    
    def minor_mat_of_mat(self,i_val,j_val):
        '''Returns the minor's matrix of a matrix at a given entry. Entries work on
            python indexing.'''
        num_rows,num_cols = self.return_mat_dim()
        i_target = i_val
        j_target = j_val
        carry_list = []
        for row in range(0,num_rows):
            for column in range(0,num_cols):
                if row != i_target and column != j_target:
                    carry_list.append(self._matrix[row][column])
        new_minor = NumMatrix()
        new_num_rows = num_rows - 1
        new_num_cols = num_cols - 1
        new_minor.build_matrix(new_num_rows,new_num_cols,carry_list)
        return new_minor
    
    def _two_by_two_det(self):
        '''If given a 2 x 2 matrix, will give determinant.'''
        if self.is_square_matrix():
            num_rows,num_cols = self.return_mat_dim()
            if num_rows == 2:
                entry_list = []
                for row in range(0,num_rows):
                    for column in range(0,num_cols):
                        entry_list.append(self._matrix[row][column])
                a = entry_list[0]
                b = entry_list[1]
                c = entry_list[2]
                d = entry_list[3]
                determinant = a*d - b*c
                return determinant
            else:
                raise RuntimeError ("Method only meant for 2 x 2 matrices.")
        else:
            raise RuntimeError ("Method only works for square matrices.")
    
    def _recursive_cofactor(self,in_matrix):
        """Works only on dimensions of 6 x 6 or smaller. Method recursively returns 
            a minor value of minor_matrix, where minor matrix is passed to this function. 
            Requires passing the NumMatrix into the method."""
        if isinstance(in_matrix,NumMatrix) and in_matrix.is_square_matrix():
            num_rows,num_cols = in_matrix.return_mat_dim()
            if num_rows <= 6:
                if num_rows == 2:
                    return in_matrix._two_by_two_det() #ad - bc
                elif in_matrix.use_simple_determinant():
                    print("Using simple determinant")
                    return in_matrix.simple_determinant(in_matrix)
                else: #All matrices bigger than 2 x 2 and not fit for simple determinant.
                    total = 0
                    for num in range(0,num_cols):
                        coeff = 0 #Initializing
                        if num%2 == 0: #Odd entries, in python it's even entries b/c 0 indexing.
                            coeff = in_matrix._matrix[0][num]
                        else: #col = 1 (2), col = 3 (4)
                            coeff = -1*in_matrix._matrix[0][num]
                        minor_mat = in_matrix.minor_mat_of_mat(0,num)
                        total += coeff*self._recursive_cofactor(minor_mat)
                    return total
            else: 
                raise RuntimeError ("This method only works on matrices that are 6 x 6 or smaller.")
        else:
            raise RuntimeError ("Either the passed argument was not a NumMatrix or the matrix in " \
            "question was not a square matrix.")
    
    def matrix_determinant(self):
        '''Returns the determinant of a matrix up to a size of 6 x 6'''
        return self._recursive_cofactor(self)
    
    def use_simple_determinant(self):
        '''Boolean that returns True if simple determinant should be used,
            and False if it should not be.'''
        if self.is_diagonal_no_zeros(): #Non-zero diags, Up-Tri, Low-Tri
            if self.is_upper_triangle_mat() or self.is_lower_triangle_mat() or self.is_diagonal_matrix():
                return True
            else:
                return False
        else: #Can still be a diagonal matrix.
            if self.is_diagonal_matrix():
                return True
            else:
                return False
    
    def simple_determinant(self,in_matrix):
        '''Returns a simple determinant or a determinant to
            a diagonal, upper-triangular, or lower-triangular
            matrix.'''
        if isinstance(in_matrix,NumMatrix):
            num_rows,num_cols = in_matrix.return_mat_dim()
            prod = 1
            for num in range(0,num_rows):
                prod *= in_matrix._matrix[num][num]
            return prod
        else:
            raise TypeError ("Can only have NumMatrices passed into it.")
    
    def matrix_product(self,other):
        """This produces a matrix product between two matrices."""
        if isinstance(other,NumMatrix):
            m_rows,r_cols = self.return_mat_dim()
            r_rows,n_cols = other.return_mat_dim()
            if r_cols == r_rows:
                new_dim_prod = m_rows*n_cols #How many entries new matrix will have.
                dot_list,vec_list1,vec_list2 = [],[],[]
                for row in self._matrix: #m columns of 1 x n row vectors
                    new_row = NumVector()
                    new_row.build_vector(1,r_cols,row)
                    vec_list1.append(new_row)
                for j_val in range(0,n_cols): #r columns of n x 1 column vectors
                    new_col = NumVector()
                    temp_col = other.retrieve_col(j_val) #[[a],[b],...,[n]] n items
                    len_tc = len(temp_col)
                    transfer_list = [temp_col[x][0] for x in range(0,len_tc)]
                    #Converted [[a],[b],...,[n]] --> [a,b,...,n] Vector can be built now.
                    new_col.build_vector(r_rows,1,transfer_list)
                    vec_list2.append(new_col)
                #list of row vectors and list of column vectors are now created.
                for row_vector in vec_list1: #vector object, m-many
                    for col_vector in vec_list2: #vector object, n-many
                        if isinstance(row_vector,NumVector) and isinstance(col_vector,NumVector):
                            #This was done for sake of accessing attributes. They are always NumVectors.
                            dot_val = row_vector.dot_product(col_vector)
                            dot_list.append(dot_val)
                ret_matrix = NumMatrix()
                ret_matrix.build_matrix(m_rows,n_cols,dot_list)
                return ret_matrix
            else:
                raise RuntimeError ("The two matrices do not have the appropriate dimensions.")
        else:
            raise TypeError ("Matrix product only works between two NumMatrix objects.")

    def set_entry_to(self,i_val,j_val,new_entry):
        """Takes a new entry and sets the value at position (i_val,j_val) 
        to the new entry. Works in typical matrix indexing manner."""
        self._matrix[i_val-1][j_val-1] = new_entry