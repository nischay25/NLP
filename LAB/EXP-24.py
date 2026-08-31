def recognize_dialog_act(sentence):

    sentence = sentence.lower()

    if sentence.endswith("?"):
        return "Question"

    elif any(word in sentence for word in ["hello", "hi", "hey"]):
        return "Greeting"

    elif any(word in sentence for word in ["please", "can you", "could you"]):
        return "Request"

    elif any(word in sentence for word in ["bye", "goodbye", "see you"]):
        return "Goodbye"

    else:
        return "Statement"


conversation = []

print("Enter conversation sentences.")
print("Type 'exit' to stop.\n")

while True:
    sentence = input("Sentence: ")

    if sentence.lower() == "exit":
        break

    act = recognize_dialog_act(sentence)

    conversation.append((sentence, act))


print("\nDialog Act Results:")

for sentence, act in conversation:
    print(f"{sentence} -> {act}")
