'''
Author: Drake Galvan Smith
DOC: 12.13.2025
Last Modified: 12.21.2025
Purpose: To create a vector class object that can be
    used to ease the creation of a matrix class.
'''

#numvector.py

class NumVector:
    """Vector class objects are meant to emulate mathematical vector
        behavior."""
    
    def __init__(self):
        """Initializes a NumVector object."""
        self._vector = []
        self._row_dim = None
        self._col_dim = None
    #Completed
    def __add__(self,other):
        if isinstance(other,NumVector):
            if self.return_dim() == other.return_dim():
                row_N,col_N = self.return_dim()
                if self.is_row_vector(): #[[a,b,...,n]]
                    lenth_self = len(self._vector[0])
                    shell_vec = []
                    for num in range(0,lenth_self):
                        entry_sum = self._vector[0][num] + other._vector[0][num]
                        shell_vec.append(entry_sum)
                    new_vector = NumVector()
                    new_vector.build_vector(row_N,col_N,shell_vec)
                    return new_vector
                else: #Is a column vector
                    pass
            else:
                raise RuntimeError ("Vectors do not have the same dimensions.")
        else:
            raise TypeError ("Can only add vector objects.")
    #Completed
    def __mul__(self,other):
        if isinstance(other,(int,float)):
            row_N,col_N = self.return_dim()
            scaled_vector = NumVector()
            vector_list = []
            if self.is_row_vector():
                for column in range(0,col_N):
                    entry_self = self._vector[0][column]
                    vector_list.append(other*entry_self)
                scaled_vector.build_vector(row_N,col_N,vector_list)
                return scaled_vector
            else: #Is a column vector
                for row in range(0,row_N): #Each row el = [a]
                    entry_self = self._vector[row][0]
                    vector_list.append(other*entry_self)
                scaled_vector.build_vector(row_N,col_N,vector_list)
                return scaled_vector
        else:
            raise TypeError ("Can only multiply scalars against matrices with this operator.")
    #Completed
    def __rmul__(self,other):
        return self.__mul__(other)
    #Completed
    def __sub__(self,other):
        pass
        if isinstance(self,NumVector) and isinstance(other,NumVector):
            return self + (-1 * other)
        else:
            raise TypeError ("Can only subtract vector objects.")
    #Completed
    def user_build_vector(self):
        """Allows the user to build a matrix through individual value
            assignment."""
        row_count = int(input("# of Rows: "))
        col_count = int(input("# of Cols: "))
        dim_prod = row_count*col_count
        if dim_prod == row_count or dim_prod == col_count:
            self._row_dim = row_count
            self._col_dim = col_count
            for row_elem in range(0,row_count):
                row_list = []
                self._vector.append(row_list)
            for row_elem in self._vector:
                for val in range(0,col_count):
                    entry = float(input("Entry: "))
                    row_elem.append(entry) #row_lists
        else:
            raise RuntimeError ("Vectors can only be row or column vectors.")
    #Completed  
    def build_vector(self,row_count:int,col_count:int,build_list:list):
        """Takes in a list. Checks if the dimensionality matches
            with that of the desired vector. If there is a match,
            list is parsed to make a vector of the given dimensions."""
        dim_prod = row_count*col_count
        if dim_prod == row_count or dim_prod == col_count:
            list_length = len(build_list)
            counter = 0
            if list_length == dim_prod:
                self._row_dim = row_count
                self._col_dim = col_count
                if row_count == 1: #Row Vector [[a,b,...,n]]
                    self._vector.append(build_list)
                else: #Column Vector [[a],[b],...,[n]]
                    for element in build_list:
                        self._vector.append([element])
            else:
                raise RuntimeError ("The number of list entries does not match the "\
                "vector dimensionality.")
        else:
            raise RuntimeError ("Vectors can only be row or column vectors.")
    #Completed
    def return_dim(self):
        """Returns vector dimensions. #Row x #Col"""
        num_rows = self._row_dim
        num_cols = self._col_dim
        return [num_rows,num_cols]
    #Completed
    def vec_entry_at(self,i_val,j_val):
        """Allows for retrieval at typical vector entries."""
        row_N,col_N = self.return_dim()
        if i_val > row_N or j_val > col_N:
            raise IndexError ("Out of vector index bounds.")
        else:
            if row_N == 1: #Row Vector
                return self._vector[0][j_val-1]
            else: #Column Vector
                return self._vector[i_val-1][0]
    #Completed
    def is_row_vector(self):
        """Boolean function the returns true if vector is a row vector and
            returns False if it isn't a row vector."""
        if self._row_dim == 1:
            return True
        else:
            return False
    #Completed
    def is_col_vector(self):
        """Boolean function the returns true if vector is a column vector and
            returns False if it isn't a column vector."""
        if self._col_dim == 1:
            return True
        else:
            return False
    #Completed
    def vector_transpose(self):
        '''Method performs a vector transpose on vector objects.'''
        #Row Vector --> Column Vector
        if self.is_row_vector():
            #Row Vec [[a,b,...,n]]
            temp_list = self._vector[0]
            column_vector = []
            for element in temp_list:
                column_vector.append([element])
            self._vector = column_vector
            hold = self._row_dim
            self._row_dim = self._col_dim
            self._col_dim = hold
        #Column Vector --> Row Vector
        else: #Is a column vector
            #Col Vec [[a],[b],...,[n]]
            temp_list = self._vector
            row_vector = []
            shell_vector = []
            for element in temp_list:
                shell_vector.append(element[0])
            row_vector.append(shell_vector)
            self._vector = row_vector
            hold = self._row_dim
            self._row_dim = self._col_dim
            self._col_dim = hold
    #Completed
    def dot_product(self,other):
        '''Performs a dot product between a row and column vector.
            Exclusively coded for that form.'''
        if isinstance(other,NumVector):
            #Row: [[a,b,...n]] [0] = [a,b,...,n]
            #Column: [[a],[b],...,[n]] [i] = [[a],[b],...,[n]]
            if len(self._vector[0]) == len(other._vector):
                if self.is_row_vector() and other.is_col_vector():
                    other.vector_transpose() #now a row vector, for sake of calculations
                    total = 0
                    for num in range(0,len(self._vector[0])):
                        #Elementwise multiplication.
                        prod = self._vector[0][num]*other._vector[0][num]
                        total += prod
                    other.vector_transpose() #Restores the vector to original form.
                    return total
                else:
                    raise RuntimeError ("This operation can only be done between a row vector" \
                    " and a column vector, in that order.")
            else:
                raise RuntimeError ("These vectors do not have the same number of elements.")
        else:
            raise TypeError ("This operation is only performed between vectors.")
