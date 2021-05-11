def removeElement(nums, val):
    i, j = 0, 0
    while j < len(nums):
        print(i, j, nums)
        if nums[i] == val:
            while nums[j] == val:
                j += 1
                if j == len(nums):
                    return i
            nums[i] = nums[j]
            nums[j] = val
        i += 1
        j += 1
    return i





nums = [0, 1,0, 2, 2, 2, 0, 2, 1, 3, 0, 4, 2]
# nums = [1, 2, 0, 2]
# nums = [1, 2,2,0,2]
val = 2

print(removeElement(nums, val))
print(nums)
