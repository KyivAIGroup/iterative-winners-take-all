def validMountainArray(arr):
    if len(arr) < 3:
        return False
    if arr[1] <= arr[0] or arr[-1] >= arr[-2]:
        return False
    go_up = True
    for i in range(1, len(arr)):
        # print(i, arr[i], go_up)
        if arr[i] == arr[i - 1]:
            return False
        if go_up:
            if arr[i] <= arr[i - 1]:
                go_up = False
        else:
            if arr[i] >= arr[i-1]:
                return False
    return True


arr = [3, 5, 5, 1]
# arr = [0,3,2,1,0]
# arr = [0,1,2,3,4,5,6,7,8,9]
# arr = [9,8,7,6,5,4,3,2,1,0]
print(validMountainArray(arr))