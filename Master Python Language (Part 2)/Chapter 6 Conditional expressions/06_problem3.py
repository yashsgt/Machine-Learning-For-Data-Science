# A spam comment is defined as a text containing following keywords: "Make a lot of money", "buy now", "subscribe this",
#  "click this". WAP to detect these spams ?

p1 = "Make a lot of money"
p2 = "buy now"
p3 = "subscribe this"
p4 = "click this"

message = input("Enter your comment : ")

if((p1 in message) or (p2 in message) or (p3 in message) or (p4 in message)):
    print("this cooment is a spam")
else:
    print("this is not a spam")

    