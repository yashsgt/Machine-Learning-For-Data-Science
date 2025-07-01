# WAP to greet all the person names stored in list 'l' and which start with S
# l = ["Yash", "Sohan", "Sachin", "Rahul"]


l = ["Yash", "Sohan", "Sachin", "Rahul"]

for name in l:
    if(name.startswith("S")):
        print(f"Hello {name}")