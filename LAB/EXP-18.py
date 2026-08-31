import re

expression = input("Enter logical expression: ")

pattern = r"(\w+)\((\w+)\)"

match = re.match(pattern, expression)

if match:
    predicate = match.group(1)
    argument = match.group(2)

    print("\nValid FOPC Expression")
    print("Predicate:", predicate)
    print("Argument:", argument)
else:
    print("Invalid Expression")
