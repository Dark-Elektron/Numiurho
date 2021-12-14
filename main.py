# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    ra = 5
    x = 3
    theta = np.linspace(0, np.pi, 10)
    r = np.sqrt(ra**2 + x**2 - 2*ra*x*np.cos(theta))
    print(list(r))
    # r = np.flip(r)

    x = r*np.cos(np.flip(theta))
    y = r*np.sin(np.flip(theta))
    print(list(np.flip(theta)))
    print(list(x))
    print(list(y))

    x1 = ra*np.cos(theta)-3
    y1 = ra*np.sin(theta)

    plt.scatter(x, y)
    plt.scatter(x1, y1)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import numpy as np

    print(np.exp(-1248))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
