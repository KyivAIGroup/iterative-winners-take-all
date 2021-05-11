def removeDuplicates(nums):
    i, j = 0, 0
    while j < len(nums):
        print(i, j, nums)
        while nums[j] == nums[i]:
            j += 1
            if j == len(nums):
                return i + 1
        nums[i + 1] = nums[j]
        i += 1
        j += 1
    return i + 1





nums = [0, 1,0, 2, 2, 2, 0, 2, 1, 3, 0, 4, 2]
nums = [0,0,1,1,1,2,2,3,3,4]
nums = [0, 0, 1,1,1, 2]
nums = [1,1,2]
# nums = [1, 2, 0, 2]
# nums = [1, 2,2,0,2]

print(removeDuplicates(nums))
print(nums)
