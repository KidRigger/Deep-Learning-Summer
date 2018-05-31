import numpy as np

# Convenience of usage. You don't need to use '' anymore.
X = 'X'
O = 'O'
E = '_'

# 2D numpy array as the TicTacToe grid.
grid = np.array([[E,E,E],[E,E,E],[E,E,E]])

# Printing utility function
def print_():
    for i in grid:
        for j in i:
            print(j,end=' ')
        print()

# Evaluates if all values in arr are same
def eval(arr):
    for i in arr:
        if i != arr[0]:
            return False
    return True

# Checks if any wins exist and returns the winner
def win():
    # Evaluates ever row in grid.
    for row in grid:
        if eval(row) and row[0] != E:
            return row[0]
    # Every row is a column
    grid2 = grid.transpose()
    # For every 'column' in original grid
    for row in grid2:
        if eval(row) and row[0] != E:
            return row[0]
    # short diagonal
    row = (grid[0,0],grid[1,1],grid[2,2])
    if eval(row) and row[0] != E:
        return row[0]
    # long diagonal
    row = (grid[0,2],grid[1,1],grid[2,0])
    if eval(row) and row[0] != E:
        return row[0]
    # if not a win
    return None

# Flips the player between X and O
def other(var):
    if var == X:
        return O
    return X

# Game starts with X
turn = X

# 10 iterations, ends after 9th (Tie)
for tcount in range(10):

    # print the grid
    print_()

    # Ran out of moves
    if tcount == 9:
        print("It's a tie!")
    else:
        # 9 turns not over
        x = win()

        # If no one won
        if x == None:
            inp = int(input("Enter a cell number: ").strip())
            while grid[inp//3,inp%3] != E:
                inp = int(input("Invalid input, try again: ").strip())
            grid[inp//3,inp%3] = turn
            turn = other(turn)

        # If someone won
        else:
            print(x + " won!")
            break
