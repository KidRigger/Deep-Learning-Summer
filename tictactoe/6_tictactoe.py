"""
#s = int(input())

def palindrome(str):

    l = len(str)
    flag = 0
    for i in range(int(l / 2)):
        if(str[i] != str[l - 1 - i]):
            flag = 1
            break

    if (flag == 0):
        print('Yes')
    else:
        print('Nope')

def prime(num):

    flag = 0
    for i in range(2, int(num / 2)):
        if(num % i == 0):
            flag = 1
            break
    if (flag == 0):
        print('Yes')
    else:
        print('No')

prime(s)
"""

grid = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]


def winner(*mat):
    flag = 0
    win = '-'
    for i in range(3):
        if((mat[0][i] == mat[1][i]) and (mat[1][i] == mat[2][i]) and (mat[0][i] != win)):
            flag = 1
            win = mat[0][i]
            break

        if((mat[i][0] == mat[i][1]) and (mat[i][1] == mat[i][2]) and (mat[i][0] != win)):
            flag = 1
            win = mat[i][0]
            break

        if((mat[0][0] == mat[1][1]) and (mat[1][1] == mat[2][2]) and (mat[0][0] != win)):
            flag = 1
            win = mat[1][1]
            break

        if((mat[2][0] == mat[1][1]) and (mat[1][1] == mat[0][2]) and (mat[1][1] != win)):
            flag = 1
            win = mat[1][1]
            break

    if (flag == 1):
        print('player ' + win + ' wins')
        exit()

for i in range(9):
    ch = True
    while ch:
        num = int(input())
        one = int(num / 3)
        two = num - (one * 3)
        if grid[one][two]!='-':
            print('Already used. Re-enter ')
            continue
        else:
            ch = False
        if (i % 2 == 0):
            grid[one][two] = 'O'
        else:
            grid[one][two] = 'X'

    print(grid[0])
    print(grid[1])
    print(grid[2])
    print()
    winner(*grid)
print('Draw!')    
exit() 
