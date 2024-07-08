# Wypisz liczby od 1 do 100, przy czym liczby podzielne przez 3 zastąp słowem ‘Fizz’, 
# liczby podzielne przez 5, zastąp słowem ‘Buzz’,
# natomiast liczby podzielne i przez 3 i przez 5 zastąp
# słowem ‘FizzBuzz’.

print("Fizz Buzz")

def fizz_buzz(n):
    result = []
    for i in range(1, n+1):
        if i % 3 == 0 and i % 5 == 0:
            result.append('FizzBuzz')
        elif i % 3 == 0:
            result.append('Fizz')
        elif i % 5 == 0:
            result.append('Buzz')
        else:
            result.append(str(i))
    return result

numbers = fizz_buzz(100)
for number in numbers:
    print(number)