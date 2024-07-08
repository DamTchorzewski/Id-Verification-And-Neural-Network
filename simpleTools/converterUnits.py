# Konwerter jednostek: Napisz program, który konwertuje różne jednostki miar 
# (np. kilometry na mile, celsjusze na farenheity).

print("Converter Units")

def miles_to_kilometers(miles):
    """Mile to kilometers converter."""
    return miles * 1.60934

def kilometers_to_miles(kilometers):
    """Kilometers to miles converter."""
    return kilometers / 1.60934

def celsius_to_fahrenheit(celsius):
    """Celsius to Fahrenheit converter."""
    return celsius * 9.0 / 5.0 + 32

def fahrenheit_to_celsius(fahrenheit):
    """Fahrenheit to Celsius converter."""
    return (fahrenheit - 32) * 5.0 / 9.0

# Example use of the converter
print("2 mile to ", miles_to_kilometers(2), "kilometers")
print("132.2 kilometers to ", kilometers_to_miles(132.2), "miles")
print("32 celsius to ", celsius_to_fahrenheit(32), "fahrenheit")
print("83 fahrenheit to ", fahrenheit_to_celsius(83), "celsius")