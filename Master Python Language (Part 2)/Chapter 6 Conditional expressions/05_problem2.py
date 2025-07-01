# WAP to find out whether a student has passed or failed if it requires a total of 40% and atleast 33% in each subject to pass.
#  Assume three subjects and take marks as a input from the user ?

a = int(input("Enter Physics marks : "))
b = int(input("Enter Chemistry marks : "))
c = int(input("Enter Maths marks : "))

total_percentage = (a + b + c)/3

if(total_percentage>=40 and a>=33 and b>=33 and c>=33):
    print("you have passed", total_percentage)
else:
    print("You are failed, try again next year", total_percentage)

