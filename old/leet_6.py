def replaceElements(arr):
    max_value = arr[-1]
    for i in range(len(arr) - 2, -1, -1):
        print(i, arr[i])
        deleted = arr[i]
        arr[i] = max_value
        if deleted > max_value:
            max_value = deleted
    arr[-1] = -1
    return True


arr = [17,18,5,4,6,1]
arr = [1]
arr = [400, 0]
# arr = [0,1,2,3,4,5,6,7,8,9]
# arr = [9,8,7,6,5,4,3,2,1,0]
print(replaceElements(arr))
print(arr)