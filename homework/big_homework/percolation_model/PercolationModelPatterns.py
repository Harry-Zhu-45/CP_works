"""
This file contains functions to add various patterns to the percolation model
"""
from PercolationModel import PercolationModel2D


def add_block(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a 2x2 block into the system, with bottom left corner (icentre, jcentre), invariant
    like this:
        xx\n
        xx\n
    """
    extent = 2
    cell.clear(icentre, jcentre, extent)                 # clear the space
    cell.grid[icentre:icentre+2, jcentre:jcentre+2] = 1  # add the block


def add_beehive(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a beehive into the system, with (icentre, jcentre) being the inner left blank square, invariant
    like this:
        oxxo\n
        xoox\n
        oxxo\n
    """
    extent = 7
    cell.clear(icentre, jcentre, extent)
    cell.grid[icentre-1, jcentre:jcentre+2] = 1  # top row
    cell.grid[icentre+1, jcentre:jcentre+2] = 1  # bottom row
    cell.grid[icentre, jcentre-1] = 1            # left dot
    cell.grid[icentre, jcentre+2] = 1            # right dot


def add_blinker(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a horizontal line of 3 blocks, a period 2 oscillator
    like this:
        xxx\n
    """
    extent = 4
    cell.clear(icentre, jcentre, extent)
    cell.grid[icentre, jcentre-1:jcentre+2] = 1


def add_loaf(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a loaf, invariant
    like this:
        ooxo\n
        oxox\n
        xoox\n
        oxxo\n
    """
    cell.grid[icentre, jcentre+1:jcentre+3] = 1
    cell.grid[icentre+1:icentre+3, jcentre+3] = 1
    cell.grid[icentre+1, jcentre] = 1
    cell.grid[icentre+2, jcentre+1] = 1
    cell.grid[icentre+3, jcentre+2] = 1


def add_boat(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a boat, invariant
    like this:
        xxo\n
        xox\n
        oxo\n
    """
    extent = 4
    cell.clear(icentre, jcentre, extent)

    indices = cell.getVonNeumannNeighbourhood(icentre, jcentre)
    for element in indices:
        cell.grid[element[0], element[1]] = 1

    cell.grid[icentre+1, jcentre-1] = 1


def add_toad(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a toad, a period 2 oscillator
    like this:
        xo\n
        xx\n
        xx\n
        ox\n
    """
    extent = 3
    cell.clear(icentre, jcentre, extent)
    cell.grid[icentre-1:icentre+2, jcentre] = 1
    cell.grid[icentre:icentre+3, jcentre-1] = 1


def add_beacon(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add two 2x2 blocks, a period 2 oscillator
    like this:
        xxoo\n
        xxoo\n
        ooxx\n
        ooxx\n
    """
    extent = 3
    cell.clear(icentre, jcentre, extent)
    add_block(cell, icentre+2, jcentre)
    add_block(cell, icentre, jcentre+2)


def add_pulsar(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a pulsar, a period 3 oscillator
    """
    extent = 8
    cell.clear(icentre, jcentre, extent)

    # Start with inner cross
    # # North
    cell.grid[icentre+2:icentre+5, jcentre+1] = 1
    cell.grid[icentre+2:icentre+5, jcentre-1] = 1

    # # South
    cell.grid[icentre-4:icentre-1, jcentre+1] = 1
    cell.grid[icentre-4:icentre-1, jcentre-1] = 1

    # # East
    cell.grid[icentre+1, jcentre+2:jcentre+5] = 1
    cell.grid[icentre-1, jcentre+2:jcentre+5] = 1

    # # West
    cell.grid[icentre+1, jcentre-4:jcentre-1] = 1
    cell.grid[icentre-1, jcentre-4:jcentre-1] = 1

    # Now do surrounding bars - quadrant at a time
    cell.grid[icentre+6, jcentre+2:jcentre+5] = 1
    cell.grid[icentre+2:icentre+5, jcentre+6] = 1

    cell.grid[icentre-4:icentre-1, jcentre+6] = 1
    cell.grid[icentre-6, jcentre+2:jcentre+5] = 1

    cell.grid[icentre-6, jcentre-4:jcentre-1] = 1
    cell.grid[icentre-4:icentre-1, jcentre-6] = 1

    cell.grid[icentre+2:icentre+5, jcentre-6] = 1
    cell.grid[icentre+6, jcentre-4:jcentre-1] = 1


def add_glider(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a glider, a period 4 oscillator
    like this:
        oxo\n
        oox\n
        xxx\n
    """
    cell.grid[icentre, jcentre:jcentre+3] = 1
    cell.grid[icentre+1, jcentre+2] = 1
    cell.grid[icentre+2, jcentre+1] = 1


def add_spaceship(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a lightweight spaceship, a period 4 oscillator
    like this:
        oxoox\n
        xoooo\n
        xooox\n
        xxxxo\n
    """
    cell.grid[icentre, jcentre:jcentre+4] = 1
    cell.grid[icentre:icentre+3, jcentre] = 1
    cell.grid[icentre+3, jcentre+1] = 1
    cell.grid[icentre+3, jcentre+4] = 1
    cell.grid[icentre+1, jcentre+4] = 1


def add_glider_gun(cell: PercolationModel2D, icentre: int, jcentre: int):
    """
    Add a Gosper glider gun, a period 30 oscillator
    """
    add_block(cell, icentre+5, jcentre+2)

    # Make first of inner patterns
    cell.grid[icentre+4:icentre+7, jcentre+12] = 1
    cell.grid[icentre+3, jcentre+13] = cell.grid[icentre+7, jcentre+13] = 1

    # cell.grid[icentre+2,jcentre+14] = cell.grid[icentre+8,jcentre+14] = 1
    cell.grid[icentre+2, jcentre+14:jcentre+16] = cell.grid[icentre+8, jcentre+14:jcentre+16] = 1
    cell.grid[icentre+5, jcentre+16] = 1
    cell.grid[icentre+3, jcentre+17] = cell.grid[icentre+7, jcentre+17] = 1
    cell.grid[icentre+4:icentre+7, jcentre+18] = 1
    cell.grid[icentre+5, jcentre+19] = 1

    # Now second pattern
    cell.grid[icentre+6:icentre+9, jcentre+22:jcentre+24] = 1
    cell.grid[icentre+5, jcentre+24] = cell.grid[icentre+9, jcentre+24] = 1
    cell.grid[icentre+4:icentre+6, jcentre+26] = cell.grid[icentre+9:icentre+11, jcentre+26] = 1

    add_block(cell, icentre+7, jcentre+36)
