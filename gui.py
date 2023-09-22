from src.train import Model

model = Model()
model.load_model("./models/model-1", train=True)

user_id = -1
while user_id < 0 or user_id > 1000:
    user_id = int(input("Enter an user id number between 1 to 1000: "))

while True:
   print("Here are some recommendation for that user: ")
   print(model.predict(user_id, 10))
   print("-"*40)
   user_id = -1
   while user_id < 0 or user_id > 1000:
    user_id = int(input("Enter an user id number between 1 to 1000: "))