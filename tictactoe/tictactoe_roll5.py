arr=[[0,0,0],
	  [0,0,0],
	 [0,0,0]]
i=0
won=0
while i<=8 and won==0  :
	num=input("enter a number")
	row=int(num/3)
	col=num-row*3
	if i%2==0 :
		if (arr[row][col] != 0) :
			print("error")
		else :
			arr[row][col]=1
	else :
		if (arr[row][col] != 0) :
			print("error")
		else :
			arr[row][col]=2
	print(arr[0])
	print
	print(arr[1])
	print
	print(arr[2])
	i=i+1
	x=0
	y=0
	win=0
	while x<3 :
		if(arr[x][y]==arr[x][y+1] and arr[x][y+1]==arr[x][y+2] and arr[x][y]==1 and win==0) :
			print("1 has won hor")
			won=1
		if(arr[x][y]==arr[x][y+1] and arr[x][y+1]==arr[x][y+2] and arr[x][y]==2 and win==0) :
			print("2 has won hor")
			won=1
		x=x+1
	x=0
	while y<3 :
	        if(arr[x][y]==arr[x+1][y] and arr[x+1][y]==arr[x][y] and arr[x+2][y]==1 and win==0) :
			print("1 has won ver")
			won=1
		if(arr[x][y]==arr[x+1][y] and arr[x+1][y]==arr[x][y] and arr[x+2][y]==2 and win==0) :
			print("1 has won ver")
			won=1

		y=y+1
	if(arr[0][0]==arr[1][1] and arr[1][1]==arr[2][2] and arr[2][2]==1 and win==0) :
		print("1 has won diag")
		won=1
	if(arr[0][0]==arr[1][1] and arr[1][1]==arr[2][2] and arr[2][2]==2 and win==0) :
		print("1 has won diag")
		won=1
	if(arr[0][2]==arr[1][1] and arr[1][1]==arr[2][0] and arr[1][1]==1 and win==0) :
		print("1 has won diag")
		won=1
	if(arr[0][2]==arr[1][1] and arr[1][1]==arr[2][0] and arr[1][1]==2 and win==0) :
		print("1 has won diag")
		won=1