# -*- coding = utf-8 -*-

#list_num = raw_input().split(' ')
# def devi():
#     a = int(list_num[0])
#     b = int(list_num[1])
#     if a % b == 0:
#         print("YES")
#     else:
#         print("NO")
#devi()

# num = input()
# lst = []
# def devision():
#     for i in xrange(num):
#         if num % (i + 1) == 0:
#             lst.append(i+1)
#     if lst == [1, num]:
#         print("Y")
#     else:
#         print("N")
# devision()

# n = input()
# def fibonacci(n):
#     if n <= 1:
#         return n
#     fibo = fibonacci(n - 1) + fibonacci(n - 2)
#     return fibo
# fibo = fibonacci(n)
# print(fibo)

# import numpy as np
# list = input().split(' ')
# num_
# matrix = np.zeros((int(num_list[0], int(num_list[1]))))

# -*- coding: utf-8 -*- 
from itertools import combinations

class Solution(object):
    # def __init__(self, nums, target):
    #     self.nums = nums
    #     self.target = target

    def twoSum(self, nums, target):
        for combins in combinations(nums,2):
            if sum(combins) == target:
                index_0 = nums.index(combins[0])
                index_1 = nums.index(combins[1])
                if index_0 != index_1:
                    return [index_0, index_1]

# class Solution(object):
#     """
#     :type nums: List[int]
#     :type target: int
#     :rtype: List[int]
#     """
#     def twoSum(self, nums, target):
#         length = len(nums)
#         for i in range(length):
#             for j in range(length - i - 1):
#                 j = i + j + 1
#                 if i == length - 1:
#                     break
#                 a_1 = nums[i]
#                 a_2 = nums[j]
#                 print(i, a_1)
#                 print(j, a_2)
#                 if a_1 + a_2 == target:
#                     res = [i, j]
#                     return res
# more pythonic
class Solution_0(object):
    def twoSum(self, nums, target):
        for i, val in enumerate(nums):
            diff = target - val
            if diff in nums:
                j = nums.index(diff)
                if i != j:
                    return [i, j]

    def twoSum_x(self, nums, target):
        hashed = {}
        for i in range(len(nums)):
            if target-nums[i] in hashed: return [hashed[target-nums[i]], i]
            hashed[nums[i]] = i

class Solution_1(object):
    def twoSum(self, nums, target):
        d = dict()
        for i, val in enumerate(nums):
            diff = target -val
            print('i=%d'%i, 'val=%d'%val, 'diff=%d'%diff)
            if diff in d:
                print('-'*15)
                print(d)
                return [d[diff], i]
            d[val] = i


if __name__ == '__main__':
    nums = [5,6,3,9,8,4,3,2]
    target = 6
    ss = Solution_1()
    a = ss.twoSum(nums, target)
    print(a)

count = 0
id(count)
def function():
    count = 1
    print(id(count))
    print(count)
    count = count + 1
    print(id(count))
    print(count)
id(count)
print(count)


# --------------------------------------------------------
# 从排序数组中删除重复项
class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        cache = set()
        for i in nums:
            cache.add(i)
        return list(cache)

class Solution_1(object):
    def removeDuplicates(self, nums):
        if len(nums) <= 1:
            return len(nums)
        i = 1
        index = 1
        while i < len(nums):
            if nums[i] != nums[i-1]:
                nums[index] = nums[i]
                index += 1
            i += 1
        return index

class Solution_2:  
    def removeDuplicates(self, nums):  
        """ 
        :type nums: List[int] 
        :rtype: int 
        """  
        i = 0  
        while i < len(nums)-1:  
            if nums[i] == nums[i+1]:  
                nums.remove(nums[i])  
            else:  
                i += 1  
        return len(nums) 


if __name__ == '__main__':
    nums = [1,2,3,4,4,5,6,6]
    s = Solution_2()
    s.removeDuplicates(nums)
    print(nums)


# ----------------------------------------------------
# 买卖股票的最佳时机 II
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        for i, val in enumerate(prices):
            for j, val_ in enumerate(prices[i+1:]):
                if val < val_:
                    buy_price = val
                    val = val_
                continue


if __name__ == '__main__':
    nums = [1,2,3,4,4,5,6,6]
    s = Solution()
    s.maxProfit(nums)


# ----------------------------------------------
# 旋转数组

class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        for i in range(k):
            out = nums.pop()
            nums.insert(0, out)

        return nums

    def rotate_1(self, nums, k):
        if k > len(nums):
            return None
        nums = nums[-k:] + nums[:-k]
        return nums

if __name__ == '__main__':
    nums = [1,2,3,4,5,6,7]
    s = Solution()
    s.rotate_1(nums,3)     


# -----------------------------------------
# 只出现一次的数字

class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        bak = set()
        for num in nums:
            if num in bak:
                bak.remove(num)
            else:
                bak.add(num)
        return list(bak)[0]
    def singleNumber_1(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        a = 0
        for i in nums:
            # 异或运算
            a ^= i  
        return a
if __name__ == '__main__':
    nums = [4,1,2,1,2]
    s = Solution()
    s.singleNumber_1(nums)  

# ---------------------------------------
# 加一
class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        # plusone = digits.pop() + 1
        # if plusone >= 10:  
        #     quotient = plusone // 10
        #     remainder = plusone % 10
        #     digits.extend([quotient,remainder])
        # else:
        #     digits.append(plusone)
        # return digits
        func = lambda x:str(x)
        plusone = int(''.join(map(func, digits))) +1

        # split_ = list(str(plusone))
        # res = list(map(lambda x: int(x), split_))
        res = list(map(int, str(plusone)))
        return res

    def plusOne_1(self, digits):
        res = 0
        l = len(digits) -1
        for i, num in enumerate(digits):
            res += num * 10 ** (l - i)
        res = list(map(int, str(res+1)))
 
        return res

if __name__ == '__main__':
    nums = [1,2,3]
    s = Solution()
    s.plusOne_1(nums)  

# -------------------------------------
# 移动零

class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        # l = len(nums)
        # for i in range(l):
        #     num = nums[i]
        #     if num == 0:
        #         nums.remove(num)
        #         nums.append(num)
        #         print(nums,'-----')
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1

        return nums
if __name__ == '__main__':
    nums = [0,0,1]
    s = Solution()
    s.moveZeroes(nums)

def decorated(func):
    print(1)
    return func

def main_f():
    print(2)
a = decorated(main_f)
a()

import numpy as np
sku = np.dtype([('sku_id', np.int32), ('desc', np.str, 50), ('value', np.float)])
print(sku)

sku2 = np.dtype({'names':['sku_id', 'desc', 'value'], 'formats':['<i4', 'S50', '<f8']})

online_shop = np.array([(1, 'apple', 2.3), (2.1, 3, 5), (3, 'banana', True)], dtype=sku2)
print(online_shop)


np.cov

# --------------------------------------------------
# 有效的数独
class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        """
        def isRowColValid(one):
            num_dict = {}
            for num in one:
                if num != '.':
                    if num not in num_dict:
                        num_dict[num] = 0
                    num_dict[num] += 1

            for _, i in num_dict.items():
                if i > 1:
                    return False
            return True
        """
        def isRowColValid(one):
            num_dict = dict().fromkeys(one, 0)
            for item in one:
                if item != '.':
                    num_dict[item] += 1
                if num_dict[item] > 1:
                    return False
            return True
        
        for row in board:
            if not isRowColValid(row):
                return False
        
        boardT = list(zip(*board))
        for row in boardT:
            if not isRowColValid(row):
                return False
        
        pos = 0
        while pos < 9:
            offset = 0
            while offset < 9:
                temprow = board[pos][offset:offset+3]
                temprow.extend(board[pos+1][offset:offset+3])
                temprow.extend(board[pos+2][offset:offset+3])
                print(temprow)
                if not isRowColValid(temprow):
                    return False
                
                offset += 3
            pos += 3
        return True

if __name__ == '__main__':
    a = [["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]]

    s = Solution()
    s.isValidSudoku(a)
# -------------------------
# 反转字符串
string = 'string'
string[::-1]


# 颠倒数字

class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        flag = 1 if x>= 0 else -1

        n = str(abs(x))[::-1]
        res = int(n) * flag
        return res if res.bit_length() < 32 else 0

# 字符串中的第一个唯一字符

class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """

        for i, val in enumerate(s):

            if val not in s[i+1:] and val not in s[:i]:
                return i

            if i+1 == len(s):
                return -1
        
        return -1

    def firstUniqChar_1(self, s):
        """
        :type s: str
        :rtype: int
        : 元素消除
        """
        unique = s
        while unique:
            if unique[0] in unique[1::]:
                unique = unique.replace(unique[0], "")
            else:
                return s.find(unique[0])
        return -1 


            
if __name__ == '__main__':
    s = Solution()
    s.firstUniqChar_1('')



# ------------------------------------
# 有效的字母异同位
# all() 函数用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。
# 元素除了是 0、空、FALSE 外都算 TRUE。
# return set(s) == set(t) and all(s.count(i) == t.count(i) for i in set(s))

# string.isalnum()
# 如果 string 至少有一个字符并且所有字符都是字母或数字则返回 True,否则返回 False
# string.isalpha() http://www.runoob.com/python/python-strings.html

# 例子：验证回文


def isPalindrome(s):
    """
    :type s: str
    :rtype: bool
    """
    s = list(filter(str.isalnum, s.lower()))
    return True if s == s[::-1] else False

class Solution:
    def myAtoi(self, str):
        str__ = str.strip().split()
        if len(str__) >= 1:
            str_ = str__[0]
        else:
            return 0
        
        lis = []
        for i in str_:
            print("i", i)
            if i == '-' or i.isdigit() or i == '+':
                lis.append(i)
                print(lis)

                if not lis[0].isdigit() and len(lis) < 2:
                    continue
                else:
                    str_ = "".join(lis)
                    try:
                        num = int(float(str_))
                        print(num)
                        continue
                    except:
                        str_ = str_[:-1]
                        print("---", str_)
                        break

                    # if str_.isdigit():
                    #     continue
                    # else:
                    #     str_ = str_[:-1]
                    #     print("---",str_)
            else:
                break
        
        try:
            num = int(float(str_))
        except:
            return 0
        if num.bit_length() < 32:
            return num
        if num > 0:
            return 2**31-1
        if num < 0:
            return -2**31
if __name__ == "__main__":
    _str = "-13+8"
    ss = Solution()
    ss.myAtoi(_str)

# 
class Solution:
    def myAtoi(self, str):
        str = str.strip()
        if not str:
            return 0
        sum = 0
        flag = 1
        if str[0] == '-':
            str = str[1:]
            flag = -1
        elif str[0] == '+' :
            str = str[1:]
        
        for c in str:
            if c.isdigit():
                sum = sum*10 + ord(c) - ord('0') 
            else:
                break
        sum = flag * sum
        if sum<-2**31:
            sum = -2**31
        if sum>2**31-1:
            sum = 2**31-1
        return sum

# Permutations 
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        for num in nums:
            new_result = []
            for seq in result:
                print(seq)
                for i in range(len(seq)+1):
                    new_result.append(seq[:i]+[num]+seq[i:])
            result = new_result
        return result

ss = Solution()
ss.permute([1,2,3])