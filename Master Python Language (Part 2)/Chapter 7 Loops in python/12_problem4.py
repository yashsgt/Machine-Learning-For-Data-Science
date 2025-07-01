#  WAP to find whether a number is prime or not ? 

n = int(input("Enter a number : "))


if(n > 1):
    for i in range(2, n):
        if (n % i) == 0:
            print("Not prime number")
            break
    else:
        print("prime number")

else:
    print("Not a prime number")


    

    
        

  