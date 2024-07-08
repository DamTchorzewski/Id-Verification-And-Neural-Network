# Generator hasła: Utwórz program,
# który generuje losowe hasła zawierające litery, cyfry i
# symbole.

import random
import string

print("Password Generator")

def password_generator(length):
    all_characters = string.ascii_letters + string.digits + string.punctuation
    password = []

    for i in range(length):
        random_character = random.choice(all_characters)
        password.append(random_character)

    return ''.join(password)

def generate_random_password():
    length = int(input("Enter password length: "))
    if length < 8:
        print("The password length must be at least 8 characters")
    else:
        password = password_generator(length)
        print(f"Randomly generated password: {password}")

generate_random_password()