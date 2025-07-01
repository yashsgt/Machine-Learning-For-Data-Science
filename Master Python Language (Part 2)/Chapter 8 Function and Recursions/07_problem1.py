def calc_prod(a=3, b=5):
    prod = a * b
    print(prod)
    return(prod)

calc_prod(2, 4)

calc_prod()  # If we don't pass any value as arguement then the default value which function have already will be in used 





def calc_prod(a, b=7):
    prod = a * b
    print(prod)
    return(prod)

calc_prod(8)      # we can also pass single value then left other value of default will be in used
    