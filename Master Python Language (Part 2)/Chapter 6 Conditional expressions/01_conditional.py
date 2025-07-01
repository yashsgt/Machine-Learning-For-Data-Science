# if-elif-else ladder.....

a = int(input("Enter your age : "))

if(a>=18):
    print("you are above the age of consent ")
    print("it is very good for you")

elif(a<0):
    print("You are entering the invalid negative age")

elif(a == 0):
    print("You are entering 0 which is not a valid age")



else:
    print("You are below the age of consent")

print("End of program")  # This print() is not under indentaion or under any if or else or elif statement so it will be printed always.....
