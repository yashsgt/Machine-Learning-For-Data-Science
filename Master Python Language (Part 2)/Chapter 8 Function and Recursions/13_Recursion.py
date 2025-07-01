# Recursion:-> When a function calls itself repeatedly.

# Print n to 1 backward

def show(n):
    if(n==0):  # Base case :- it is used to stop recursion
        return
    print(n)
    show(n-1)
    print("Yash Rajput")

show(6)  