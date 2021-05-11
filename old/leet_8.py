def sortArrayByParity(A):
    i, j = 0, len(A) - 1
    while i < j:
        print(i, j, A)
        if A[i] % 2 == 1 and A[j] % 2 == 0:
            A[i], A[j] = A[j], A[i]
        if A[i] % 2 == 0:
            i += 1
        if A[j] % 2 == 1:
            j -= 1
    return A

nums = [3,1,2,4]
nums = [3,3,3,4]

print(sortArrayByParity(nums))
print(nums)