# WAP to find the greatest of four numbers entered by the user.....

a = int(input("Enter first number :"))
b = int(input("Enter second number :"))
c = int(input("Enter third number : "))
d = int(input("Enter fourth number : "))

if(a>b and a>c and a>d):
    print("a is the greatest number")
elif(b>c and b>a and b>d):
    print("b is the greatest number")
elif(c>a and c>b and c>d):
    print("c is the greatest number")
elif(d>a and d>b and d>c):
    print("d is the greatest number")
