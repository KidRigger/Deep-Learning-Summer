def check(board, x):
	for r in range(0,3):
		for c in range(1,4):
			i = r
			j = c
			if board[j] == board[3 + j] == board[6+ j] == x:
				return 1
			if board[3*i+1] == board[3*i+2] == board[3*i+3] == x:
				return 1

	if board[1]==board[5] == board[9] == x:
		return 1
	if board[3] == board[5] == board[7] == x:
		return 1
	return 0

def display(board):
	for i in range(0,3):
		for j in range(1,4):
			if board[3*i + j] == 0:
				print(" ", end="")
			else:
				print(board[3*i + j], end="")
			if j!=3:
				print(" | ", end="")
		if i!=2:
			print("\n--|---|--")
	print("")

board = []

for i in range(10):
	board.append(0)

board[0] = 'X'

for i in range(9):
	display(board)
	if i%2==1:
		n = 0
		while board[n] != 0:
			n = int(input("Player 2, please enter your move : "))
			if board[n] != 0:
				print("Invalid input !")
			else:
				board[n] = 'O'
				break

		if check(board, 'O'):
			display(board)
			print("Player 2 wins !!")
			exit(0)
	else:
		n = 0
		while board[n] != 0:
			n = int(input("Player 1, please enter your move : "))
			if board[n] != 0:
				print("Invalid input !")
			else:
				board[n] = 'X'
				break

		if check(board, 'X'):
			display(board)
			print("Player 1 wins !!")
			exit(0)

display(board)
print("It's a TIE !")