

while True:
    try:
        x = int(input("Please enter a number: "))
        break
    except ValueError:  # 有异常则执行
        print("Oops! That was no valid number. Try again...")


def plus():
    try:
        number1, number2 = eval(input("Enter two numbers, separated by a comma:"))  # eval是将字符串str当成有效的表达式来求值并返回计算结果
        result = number1 / number2
    except ZeroDivisionError:
        print("Division by zero!")
    except SyntaxError:
        print("A comma maybe missing in the input")
    except:
        print("Something wrong in the input")
    else:
        print("No exceptions, the result is", result)
    finally:
        print("executing the finally clause")
plus()


import math
def plus_():
    print("This program finds the real solution to a quadratic.\n")
    try:
        a,b,c = eval(input("please enter the coefficients (a, b, c): "))
        discRoot = math.sqrt(b * b - 4*a*c)
        root1 = (-b + discRoot) / (2*a)
        root2 = (-b - discRoot) / (2*a)
        print("\nThe solution are:", root1, root2)
    except ValueError as excObj:
        if str(excObj) == "math domain error":
            print("No Real Roots.")
        else:
            print("You didn't give me the right number of coefficients.")
    except NameError:
        print("\nYou didn't enter three numbers.")
    except TypeError:
        print("\nYour inputs are not all numbers.")
    except SyntaxError:
        print("\nYour input is not in the correct form. Missing a comma?")
    except:
        print("\nSomething went wrong, sorry!")
plus_()
