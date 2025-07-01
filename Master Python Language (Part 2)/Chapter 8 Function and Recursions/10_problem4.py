# WAF to print the factorial of n (n is the parameter)



def cal_fact(n):
    fact = 1
    for i in range(1, n+1):
        fact *= i
    print(fact)
    return fact

cal_fact(5)
cal_fact(6)







        

