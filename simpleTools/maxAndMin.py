# Znaleźć najmniejszą oraz największa liczbę na liście.
# Czyli, dla przykładu, jeżeli nasza lista to:
# lista = [1,4,-4,7]
# To najmniejsza liczba wynosi -4, natomiast największa 7

print("Max and Min volue")

def min_max(list):
    if not list:
        return None, None

    min_val = max_val = list[0]

    for num in list:
        if num < min_val:
            min_val = num
        elif num > max_val:
            max_val = num

    return min_val, max_val

list = [1,4,-4,7,10,15,100,-13,-2,-8]
minimum, maximum = min_max(list)

print(f"The smallest number on the list is: {minimum}")
print(f"The largest number on the list is: {maximum}")