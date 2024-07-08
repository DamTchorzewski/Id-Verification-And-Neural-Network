# Mamy podany ciąg S. Np „Ala ma kota”. W ramach zadania zróbcie poniższe punkty:
# 1. Zliczyć wyrazy. W naszym przypadku będzie ich 3
# 2. Zliczyć litery. Będzie ich 9
# 3. Zbadać częstotliwość występowania liter. a – 3, l – 1, m 1, k – 1, t – 1

import string

print("Operations String")

def count_words_letters_freq(S):
    words = S.split()
    letters = ''.join(words)
    letter_counts = {}

    for char in letters:
        if char.isalpha():
            char = char.lower()
            if char in letter_counts:
                letter_counts[char] += 1
            else:
                letter_counts[char] = 1

    return len(words), len(letters), letter_counts

S = "Ala ma kota, kot ma Ale"
num_words, num_letters, letter_counts = count_words_letters_freq(S)

print(f"The number of words in S is: {num_words}")
print(f"The number of letters in S is: {num_letters}")
print(f"Frequency of letters in S: {letter_counts}")