# Sprawdzanie liczby parzystej: Napisz program,
# który sprawdza, czy podana liczba jest
# parzysta czy nieparzysta.

print("Even Numbers")

def is_even(num):
    return num % 2 == 0

def check_even_odd(num):
    if is_even(num):
        return f"The number {num} is even"
    else:
        return f"The number {num} is odd"

num = 5
result = check_even_odd(num)
print(result)