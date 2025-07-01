# WAP to calculate the grade of a student from his marks from the following scheme : 
# 90 - 100 => Excellent
# 80 - 90 => grade A
# 70 - 80 => grade B
# 60 - 70 => grade C
# 50 - 60 => grade D
# <50 => Fail.....

marks = int(input("Enter your marks : "))

if(100>=marks>=90):
    print("grade A")
elif(90>marks>=80):
    print("grade B")
elif(80>marks>=70):
    print("grade C")
elif(70>marks>=60):
    print("grade D")

else:
    print("You are failed")
