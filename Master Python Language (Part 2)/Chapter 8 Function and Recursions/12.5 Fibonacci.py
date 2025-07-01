num_of_terms = int(input("Enter series :-> "))

first_term = 0
second_term = 1
print(first_term)
print(second_term)

for i in range(1,num_of_terms-1 ):

    third_num = first_term + second_term
    print(third_num)
    first_term = second_term
    second_term = third_num
    