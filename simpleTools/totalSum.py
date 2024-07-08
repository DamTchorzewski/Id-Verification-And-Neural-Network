# Sumowanie liczb: Napisz program, który sumuje liczby 
# wprowadzone przez użytkownika aż do wprowadzenia słowa "stop".

print("Total Sum")

# Initialize the sum
total_sum = 0

# A loop that will be executed until the word "stop" is entered
while True:
    # User enters a number
    number = input("Enter a number (or type 'stop' to finish): ")

    # Checking if the user to introduce the "stop"
    if number.lower() == "stop":
        break

    # Adding numbers to the sum
    total_sum += float(number)

# Writing out the sum
print("Sum of entered numbers:", total_sum)