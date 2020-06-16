


# move all values to top of matrix
matrix_to_write = [[0 for x in range(7)] for y in range(5)]
for i in range(len(matrix_to_write)-1, 2, -1):
    for j in range(6 - i):
        matrix_to_write[i][j] = 1
for i in range(len(matrix_to_write)):
    print(matrix_to_write[i])

print("\n\n\n")


#for k in range(len(matrix_to_write)-1, 1, -1):
for k in range(1, len(matrix_to_write)):
    for j in range(len(matrix_to_write[k])):
        if matrix_to_write[k][j] == 1:
            curr_row = k
            while(curr_row > 0 and matrix_to_write[curr_row-1][j] == 0):
                matrix_to_write[curr_row-1][j] = 1
                matrix_to_write[curr_row][j] = 0
                curr_row -= 1

for i in range(len(matrix_to_write)):
    print(matrix_to_write[i])



num_empty_rows = 0
for k in range(len(matrix_to_write)-1, -1, -1):
    no_data = True
    for j in range(len(matrix_to_write[k])):
        if matrix_to_write[k][j] != 0:
            no_data = False
            break
    if no_data:
        num_empty_rows += 1
print(num_empty_rows)

for k in range(0, len(matrix_to_write) - num_empty_rows):
    print(matrix_to_write[k])




'''

            # move all values to top of matrix
            for k in range(len(matrix_to_write)-1, 1, -1):
                for j in range(len(matrix_to_write[k])):



                    test1 = matrix_to_write[k - 1][j]
                    test2 = matrix_to_write[k][j]
                    if matrix_to_write[k-1][j] == "":
                        matrix_to_write[k-1][j] = matrix_to_write[k][j]
                        matrix_to_write[k][j] = ""
'''