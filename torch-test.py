import torch




# Define the original matrix
# original_matrix = torch.tensor([
#     [1, 2],
#     [3, 4],
#     [5, 6],
# ])
original_matrix = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])

print(original_matrix)

# Determine the number of rows and columns in the original matrix
# 
# Print the reshaped matrix
new_num_rows = 6
new_num_cols = 5

features_dict = torch.zeros(new_num_rows, new_num_cols, dtype=torch.int32)


print(features_dict)

num_rows, num_cols = original_matrix.shape

start_x = 0
start_y = 0


for k in range(5):
    for i in range(num_cols):
        for j in range(num_rows):
            if(start_y+i == new_num_cols):
                print("we are here now")
                start_x = start_x + num_rows
                start_y = -i
            features_dict[start_x+j, start_y+i] = original_matrix[j, i]
            print(features_dict)
        # start_y = start_y+i
    print(f"start: {start_x}, {start_y}")
    start_x = -j
    start_y = start_y + num_cols
    
    
    # original_matrix = (k+1)*torch.ones(3,2)
    # print(original_matrix)

    # for i in range(num_rows):
    #     for j in range(num_cols):
    #         reshaped_matrix[start_x+i, start_y+j] = original_matrix[i, j]

    #         print(reshaped_matrix)

    # print(start_x+i)
    # print(start_y+j)



# Print the reshaped matrix
print(reshaped_matrix)