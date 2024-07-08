# Bubble sort

print("Bubble Sort")

def bubble_sort(list):
    n = len(list)
    for i in range(n):
        for j in range(0, n-i-1):
            if list[j] > list[j+1]:
                list[j], list[j+1] = list[j+1], list[j]
    return list

numbers = [64, 34, 25, 12, 22, 11, 90, 3, 21, 43, 8, 87]
sorted_numbers = bubble_sort(numbers)
print("Sorted numbers: ", sorted_numbers)