import numpy as np


class PercolationModel2D(object):
    """
    Object that calculates and displays behaviour of 2D cellular automata
    """

    def __init__(self, ni):
        """
        Constructe a 2D cellular automaton
        input:
            ni: number of cells in each direction
        """
        self.N = ni                # Number of cells in each direction
        self.Ntot = self.N*self.N  # Total number of cells

        self.grid = np.zeros((self.N, self.N))
        self.nextgrid = np.zeros((self.N, self.N))
        self.tested = np.zeros((self.N, self.N))

        self.complete = False  # Boolean to indicate whether the model has completed, default is False

    def getMooreNeighbourhood(self, i, j):
        """
        Return a set of indices corresponding to the Moore Neighbourhood, i.e. the cells immediately adjacent to (i,j) and the cells diagonally adjacent to (i,j)
        input:
            `i`: row index
            `j`: column index
        output:
            `indices`: list of indices corresponding to the Moore Neighbourhood
        """
        indices = []
        for iadd in range(i-1, i+2):
            for jadd in range(j-1, j+2):
                if (iadd == i and jadd == j):
                    continue  # exclude the cell itself

                if (iadd > self.N-1):
                    iadd = iadd - self.N  # periodic boundary conditions
                if (jadd > self.N-1):
                    jadd = jadd - self.N  # periodic boundary conditions

                indices.append([iadd, jadd])
        return indices

    def getVonNeumannNeighbourhood(self, i, j):
        """
        Return a set of indices corresponding to the Von Neumann Neighbourhood, i.e. the cells immediately adjacent to (i,j)
        input:
            `i`: row index
            `j`: column index
        output:
            `indices`: list of indices corresponding to the Von Neumann Neighbourhood
        """
        indices = []
        for iadd in range(i-1, i+2):
            if (iadd == i):
                continue
            if (iadd > self.N-1):
                iadd = iadd - self.N
            indices.append([iadd, j])

        for jadd in range(j-1, j+2):
            if (jadd == j):
                continue
            if (jadd > self.N-1):
                jadd = jadd - self.N
            indices.append([i, jadd])
        return indices

    def check_complete(self):
        """
        Check if all cells have been tested
        output:
            `complete`: boolean, whether the model has completed
        """
        ntested = np.sum(self.tested)   # number of cells that have been tested
        if (ntested == self.N*self.N):  # if all cells have been tested
            self.complete = True        # indicate the model has completed
        return self.complete

    def randomise(self):
        """
        Place a random selection of zeros and ones into grid
        """
        for i in range(self.N):
            for j in range(self.N):
                self.grid[i, j] = np.rint(np.random.random())

    def randomise_with_symmetry(self):
        """
        Place a random selection of zeros and ones into grid, with centual symmetry
        """
        for i in range(self.N/2):
            for j in range(self.N/2):
                self.grid[i, j] = np.rint(np.random.random())
                self.grid[i+self.N/2, j] = self.grid[i, j+self.N/2] = self.grid[i+self.N/2, j+self.N/2] = self.grid[i, j]

    def clear(self, icentre, jcentre, extent):
        """
        Clear a space on the grid
        input:
            `icentre`: row index of centre of space to clear
            `jcentre`: column index of centre of space to clear
            `extent`: extent of space to clear
        """
        for i in range(icentre-extent, icentre+extent):
            for j in range(jcentre-extent, jcentre+extent):
                if (i > 0 and i < self.N and j > 0 and j < self.N):
                    self.grid[i, j] = 0

    def updateGrid(self):
        """
        Take the changes queued up on self.nextgrid, and applies them to self.grid
        """
        self.grid = np.copy(self.nextgrid)
        self.nextgrid = np.zeros((self.N, self.N))

    def ApplyPercolationModelRule(self, P):
        """
        Construct the self.nextgrid matrix based on the properties of self.grid
        Applie the Percolation Model Rules:
            1. Cells attempt to colonise their Moore Neighbourhood with probability P
            2. Cells do not make the attempt with probability 1-P
        """
        for i in range(self.N):
            for j in range(self.N):
                # If cell has already been tested
                if (self.tested[i, j] == 1):
                    self.nextgrid[i, j] = self.grid[i, j]  # Copy value from self.grid to self.nextgrid
                    continue                               # Skip to next cell

                # If cell contains a coloniser, then decide whether to colonise
                if (self.grid[i, j] == 1 and self.tested[i, j] == 0):
                    randtest = np.random.rand()

                    # If colonisation occurs
                    if (randtest < P):
                        self.nextgrid[i, j] = 1
                        indices = self.getMooreNeighbourhood(i, j)

                        for element in indices:
                            if (self.tested[element[0], element[1]] == 1):
                                continue
                            if (self.grid[element[0], element[1]] == 0):
                                self.nextgrid[element[0], element[1]] = 1

                    else:
                        self.nextgrid[i, j] = -1

                    self.tested[i, j] = 1
