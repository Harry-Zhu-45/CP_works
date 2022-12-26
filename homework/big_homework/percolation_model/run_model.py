import PercolationModelPatterns as game
from PercolationModel import PercolationModel2D
from numpy import log10
import matplotlib.pyplot as plt
from tqdm import trange


# Set up the model parameters
N = 100
nsteps = 1000
nzeros = int(log10(nsteps))+1
icentre = 50
jcentre = 50


# Create the percolation model, and seed four colony sites at the centre
cell = PercolationModel2D(N)
game.add_block(cell, icentre, jcentre)
P = 0.45  # Set the probability of a cell being occupied


# plot
fig = plt.figure()
ax = fig.add_subplot(111)

for istep in trange(nsteps):
    # Clear axes for next drawing
    ax.clear()

    # Draw the automaton
    hist = ax.pcolor(cell.grid, edgecolors='black', cmap='binary', vmin=-1, vmax=1)
    ax.text(0.9, 1.05, str(istep)+" Steps", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
    ax.text(0.1, 1.05, "P="+str(P), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=18)
    plt.show()  # Draw the figure
    plt.savefig('step_'+str(istep).zfill(nzeros)+".png")

    # Apply the Game of Life Rule, and update the grid
    cell.ApplyPercolationModelRule(P)
    cell.updateGrid()

    if cell.check_complete():
        print("Run complete")
        break

hist = ax.pcolor(cell.grid, edgecolors='black', cmap='binary', vmin=-1, vmax=1)

plt.show()
