def moveZeroes(nums):
    write_pointer = 0
    for i in range(len(nums)):
        # print(i, nums[i], write_pointer)
        if nums[i]:
            nums[write_pointer] = nums[i]
            write_pointer += 1
    for i in range(write_pointer, len(nums)):
        nums[i] = 0

nums = [0,1,0,3,12]
nums = [0, 0, 2]
# arr = [0,1,2,3,4,5,6,7,8,9]
# arr = [9,8,7,6,5,4,3,2,1,0]
print(moveZeroes(nums))
print(nums)