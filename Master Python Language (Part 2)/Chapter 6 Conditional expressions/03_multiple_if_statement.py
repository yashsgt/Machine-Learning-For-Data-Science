a = int(input("enter your age : "))

# if statement no: 1
if(a%2==0):
    print("a is even")
# End of if statement no: 1

# if statement no: 2
if(a>=18):
    print("you are above the age of consent ")
    print("it is very good for you")

elif(a<0):
    print("You are entering the invalid negative age")

elif(a == 0):
    print("You are entering 0 which is not a valid age")



else:
    print("You are below the age of consent")
# End of if statement no: 2

print("End of program")


# Note:-> elif and else can't be written alone there must be if statement also with them but if statement can also be written alone.....
