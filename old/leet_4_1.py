def checkIfExist(arr):
    hash = set()
    for n in arr:
        if 2 * n in hash:
            return True
        if n % 2 == 0:
            if n / 2 in hash:
                return True
        hash.add(n)
    return False

# arr = [7,1,14,11]
arr = [3,1,7,11]

print(checkIfExist(arr))



