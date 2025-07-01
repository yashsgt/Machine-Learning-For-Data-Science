# WAP to find out whether a given post is tallking about "yash" or not ?

post = input("Enter the post : ")

if("Yash".lower() in post.lower()):
    print("This post is talking about Yash")
else:
    print("This post is not talking about Yash")