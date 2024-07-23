# DSA Templates
These are some of the DSA templates I keep handy while leetcoding. Its great to have some of these memorized to focus more brain power on solving the problem instead of thinking how BFS worked again.

<details>
<summary>Binary Search + Modification Template [ Python ]</summary>

![image](https://github.com/user-attachments/assets/d958f676-5f24-42d2-8562-24ef37232aba)

```python
def binary_search(array) -> int:
    def condition(value) -> bool:
        pass

    # could be [0, n], [1, n] etc. Depends on problem
    left, right = min(search_space), max(search_space) 

    while left < right:
        mid = left + (right - left) // 2

        if condition(mid):
            right = mid
        else:
            left = mid + 1
    
    # **left is the minimal k satisfying the condition function**
    return left
```

![bs](https://github.com/user-attachments/assets/6cd6a88e-a14c-4284-b838-a1d6a87abf64)

![bs 1](https://github.com/user-attachments/assets/7fdd3fe8-9e77-4748-90cc-edc9691fa53f)

</details>

<details>
<summary>Sliding Window Technique Template [ Python ]</summary>

**Fixed Window - Method #1**

```python
# Fixed Sliding Window of length k
def fixedSlidingWindow(numbers, k):
    windowStart, rollingSum = 0, 0

    for windowEnd in range(len(numbers)):
        elementIncludedFromRight = numbers[windowEnd]
        rollingSum += elementIncludedFromRight
        
        # '>=k-1' very important, in this iteration we need to make sure
        # the window size is exactly 'k-1' so that when element is added
        # from right in next iteration, the size of window will be exactly 'k'
        if windowEnd >= k-1:
            elementExcludedFromLeft = numbers[windowStart]
            rollingSum -= elementExcludedFromLeft 
            windowStart += 1

    return rollingSum
```

**Fixed Window - Method #2**

```python
# Fixed Sliding Window of length k
def fixedSlidingWindow(numbers, k):
    windowStart = 0

    for windowEnd in range(len(numbers) - k + 1):
        # Traverse the window of length k
        for windowIndex in range(windowEnd, windowEnd + k):
            elementInsideSlidingWindow = numbers[windowIndex]
            # Your Code executed while traversing every window here
```

**Dynamic Window**

```python
# Longest Substring with K Distinct Characters
# Here 'K' is distinct characters
def dynamicSlidingWindow(string, k):
    charFreq = defaultdict(int)
    windowStart, maxLen = 0, 0

    for windowEnd, rightChar in enumerate(string):
        #Increment frequency of char
        charFreq[rightChar] += 1

        # Dynamically adjust the window from left
        # while pausing the expansion towards right 
        while len(charFreq) > K:
            leftChar = charFreq[windowStart]
            charFreq[leftChar] -= 1
        
            if charFreq[leftChar] == 0:
                del charFreq[leftChar]
            
            windowStart += 1

        maxLen = max(maxLen, windowEnd - windowStart + 1)

    return maxLen
```

</details>

<details>
<summary>Monotonic Queue [ Python ]</summary>

**Min Monotonic Queue**

```python
class MinMonotonicQueue:
    def __init__(self):
        # pq -: PriorityQueue or Heap
        self.queue = collections.deque()
        self.pq = []
        self.index = 0

    def enqueue(self, val):
        self.queue.append((val, self.index))
        heapq.heappush(self.pq, (val, self.index))
        self.index += 1

    def dequeue(self):
        val, prevIndex = self.queue[0]
        self.queue.popleft()
        
        # VERY IMP PART
        # the 'prevIndex' here depicts the ORDER in which elements were enqueued
        # Below condition says if an element with index before the currentMax is found
        # Simply pop it off, coz 'val' IS the MINIMUM till here and all elements before
        # 'val' are waste for us since we are going in a SLIDING WINDOW fasion, so elements
        # with indices on the left will be pushedoff before the right ones
        while prevIndex >= self.pq[0][1]:
            heappop(self.pq)

    def getMin(self):
        return self.pq[0][0]
```

**Max Monotonic Queue**

```python
class MaxMonotonicQueue:
    def __init__(self):
        # pq -: PriorityQueue or Heap
        self.queue = collections.deque()
        self.pq = []
        self.index = 0

    def enqueue(self, val):
        # -val because heapq only handles MinHeaps 
        # so invert the sign of element for maxHeap
        self.queue.append((-val, self.index))
        heapq.heappush(self.pq, (-val, self.index))
        self.index += 1

    def dequeue(self):
        val, prevIndex = self.queue[0]
        self.queue.popleft()
        
        # VERY IMP PART
        # the 'prevIndex' here depicts the ORDER in which elements were enqueued
        # Below condition says if an element with index before the currentMax is found
        # Simply pop it off, coz 'val' IS the MAXIMUM till here and all elements before
        # 'val' are waste for us since we are going in a SLIDING WINDOW fasion, so elements
        # with indices on the left will be pushedoff before the right ones
        while prevIndex >= self.pq[0][1]:
            heappop(self.pq)

    def getMax(self):
        return -self.pq[0][0]
```

</details>

<details>
<summary>BFS Leetcode Template [ Python ]</summary>

```python
from collections import deque

matrix = [
    [1,0,1],
    [0,1,0],
    [1,0,1]
]

def bfs(matrix):
    rows, cols = len(matrix), len(matrix[0])
    visited = set()
    directions = ((1,0), (-1,0), (0,1), (0,-1))
    
    def traverse(i, j):
        queue = deque([(i, j)])
        
        while queue:
            curr_i, curr_j = queue.popleft()
            
            if (curr_i, curr_j) not in visited:
                visited.add((curr_i, curr_j))
                for a, b in directions:
                    nexti, nextj = curr_i+a, curr_j+b
                    
                    if 0 <= nexti < rows and 0 <= nextj < cols:
                        #add your code
                        
                        queue.append((nexti, nextj))

                        
    for i in range(rows):
        for j in range(cols):
            traverse(i, j)

bfs(matrix)
```

</details>

<details>
<summary>DFS Leetcode Template [ Python ]</summary>

```python
class DFS(matrix):
    rows, col = len(matrix), len(matrix[0])
    visited = set()
    directions = ((1,0), (-1,0), (0,1), (0,-1))

    def traverse(i, j):
        if (i, j) in visited:
            return

        visited.add((i, j))

        for direction in directions:
            next_i, next_j = i + direction[0], j + direction[1]
            if 0 <= next_i < rows and 0 <= next_j < cols:
                # your code
                traverse(next_i, next_j)

    for i in rows:
        for j in cols:
            traverse(i, j)
```

</details>

<details>
<summary>Topological Sort [ Python ]</summary>

```python
graph = [[],[],[3],[1],[0,1],[0,2]]
from collections import deque

def topsort(graph):
    N = len(graph)
    inDegree = [0 for _ in range(N)]
    for i in graph:
        for j in i:
            inDegree[j] += 1
    
    queue = deque()
    print(inDegree)
    for i in range(N):
        if inDegree[i] == 0:
            queue.append(i)

    index = 0
    order = [-1 for _ in range(N)]
    print(queue)
    while queue:
        t = queue.popleft()
        order[index] = t
        index += 1
        for j in graph[t]:
            inDegree[j] -= 1
            if inDegree[j] == 0:
                queue.append(j)
    
    if index != N:
        print("cycle detected, Sort not possible")
        print(order)
        return []

    return order

sortedlist = topsort(graph)
print(sortedlist)
```

</details>

<details>
<summary>Union Find [ Python ]</summary>

```python
class UnionFind:
    def __init__(self, size):
        self.size = size
        self.numberOfComponents = size
        self.sz = [1] * size
        self.id = [i for i in range(size)]

    def find(self, p):
        root = p
        while root != self.id[root]:
            root = self.id[root]

        # Path compression
        while p != root:
            next_node = self.id[p]
            self.id[p] = root
            p = next_node

        return root

    def connected(self, p, q):
        return self.find(p) == self.find(q)

    def unify(self, p, q):
        if self.connected(p, q):
            return

        root1 = self.find(p)
        root2 = self.find(q)

        if self.sz[root1] >= self.sz[root2]:
            self.sz[root1] += self.sz[root2]
            self.id[root2] = root1
        else:
            self.sz[root2] += self.sz[root1]
            self.id[root1] = root2

        self.numberOfComponents -= 1
```

</details>

<details>
<summary>AVL Tree [ Python ]</summary>

```python
class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value
        self.height = 1

class AvlTree:
    def __init__(self):
        self.root = None

    def insert(self, node, key):
        t = None

        if node == None:
            t = Node(key)
            return t
        
        if key < node.value:
            node.left = self.insert(node.left, key)
        elif key > node.value:
            node.right = self.insert(node.right, key)

        # update node height
        node.height = self.NodeHeight(node)

        # perform rotations if needed

        # LL rotation
        if self.BalanceFactor(node) == 2 and self.BalanceFactor(node.left) == 1:
            return self.LLrotation(node) 
        # LR rotation
        if self.BalanceFactor(node) == 2 and self.BalanceFactor(node.left) == -1:
            return self.LRrotation(node) 
        # RR rotation
        if self.BalanceFactor(node) == -2 and self.BalanceFactor(node.right) == -1:
            return self.RRrotation(node) 
        # RL rotation
        if self.BalanceFactor(node) == -2 and self.BalanceFactor(node.right) == 1:
            return self.RLrotation(node) 

        return node

    def LLrotation(self, node):
        nodeA = node
        nodeB = node.left

        nodeA.left = nodeB.right
        nodeB.right = nodeA
        nodeA.height = self.NodeHeight(nodeA)
        nodeB.height = self.NodeHeight(nodeB)
        if node == self.root:
            self.root = nodeB
        return nodeB

    def RRrotation(self, node):
        nodeA = node
        nodeB = node.right

        nodeA.right = nodeB.left
        nodeB.left = nodeA
        nodeA.height = self.NodeHeight(nodeA)
        nodeB.height = self.NodeHeight(nodeB)
        if node == self.root:
            self.root = nodeB
        return nodeB

    def LRrotation(self, node):
        nodeA = node
        nodeB = node.left
        nodeC = node.left.right

        nodeB.right = nodeC.left
        nodeA.left = nodeC.right
        nodeC.left = nodeB
        nodeC.right = nodeA
        nodeA.height = self.NodeHeight(nodeA)
        nodeB.height = self.NodeHeight(nodeB)
        nodeC.height = self.NodeHeight(nodeC)
        if node == self.root:
            self.root = nodeC
        return nodeB

    def RLrotation(self, node):
        nodeA = node
        nodeB = node.right
        nodeC = node.right.left

        nodeB.left = nodeC.right
        nodeA.right = nodeC.left
        nodeC.left = nodeA
        nodeC.right = nodeB
        nodeA.height = self.NodeHeight(nodeA)
        nodeB.height = self.NodeHeight(nodeB)
        nodeC.height = self.NodeHeight(nodeC)
        if node == self.root:
            self.root = nodeC
        return nodeB

    def NodeHeight(self, node):
        leftHeight = node.left.height if node is not None and node.left is not None else 0
        rightHeight = node.right.height if node is not None and node.right is not None else 0

        return max(leftHeight, rightHeight) + 1

    def BalanceFactor(self, node):
        leftHeight = node.left.height if node is not None and node.left is not None else 0
        rightHeight = node.right.height if node is not None and node.right is not None else 0

        return leftHeight - rightHeight

def traverse(node):
    if node == None:
        return
    traverse(node.left)
    print(node.value)
    traverse(node.right)

avl = AvlTree()

root = avl.insert(None, 10)
avl.root = root
avl.insert(root, 5)
avl.insert(root, 2)
# print the nodes
traverse(avl.root)
```

</details>

<details>
<summary>Min Int Heap [ Python ]</summary>

```python
class MinIntHeap:

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.items = [0.0 for _ in range(capacity + 1)]

    def getLeftChildIndex(self, parentIndex):
        return 2 * parentIndex + 1

    def getRightChildIndex(self, parentIndex):
        return 2 * parentIndex + 2

    def getParentIndex(self, childIndex):
        return (childIndex-1)//2

    def hasLeftChild(self, index):
        return self.getLeftChildIndex(index) < self.size

    def hasRightChild(self, index):
        return self.getRightChildIndex(self,index) < self.size

    def hasParent(self, index):
        return self.getParentIndex(index) >= 0

    def leftChild(self, index):
        return self.items[self.getLeftChildIndex(index)]

    def rightChild(self, index):
        return self.items[self.getRightChildIndex(index)]

    def parent(self, index):
        return self.items[self.getParentIndex(index)]

    def swap(self, a, b):
        temp = self.items[a]
        self.items[a] = self.items[b]
        self.items[b] = temp

    def ensureExtraCapacity(self):
        if self.size == self.capacity:
            self.items = self.items + [0 for _ in range(self.capacity)]
            self.capacity *= 2

    def peek(self):
        if self.size == 0:
            # raise Exception("Heap Empty")
            return None
        return self.items[0]

    def poll(self):
        if self.size == 0:
            return None
        item = self.items[0]
        self.items[0] = self.items[self.size-1]
        self.size -= 1
        self.heapifyDown()
        return item

    def add(self, item):
        self.ensureExtraCapacity()
        self.items[self.size] = item
        self.size = self.size + 1
        self.heapifyUp()

    def addList(self, nums):
        for i in nums:
            self.add(i)
        self.printHeap()

    def heapifyUp(self):
        index = self.size-1
        # while inside heap and parent is bigger than element added
        while self.hasParent(index) and self.parent(index) > self.items[index]:
            self.swap(self.getParentIndex(index), index)
            index = self.getParentIndex(index)

    def heapifyDown(self):
        index = 0
        while self.hasLeftChild(index):
            smallerChild = self.getLeftChildIndex(index)
            if self.leftChild(index) > self.rightChild(index):
                smallerChild = self.getRightChildIndex(index)

            if self.items[index] > self.items[smallerChild]:
                self.swap(index, smallerChild)

            index = smallerChild

    def printHeap(self):
        for i in self.items:
            print(str(i) + "  ", end="")
        print("")
```

</details>

<details>
<summary>Max Int Heap [ Python ]</summary>

```python
class MaxIntHeap: 

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.items = [0 for _ in range(capacity + 1)]

    def getLeftChildIndex(self, parentIndex):
        return 2 * parentIndex + 1
    def getRightChildIndex(self, parentIndex):
        return 2 * parentIndex + 2
    def getParentIndex(self, childIndex):
        return (childIndex-1)//2
    
    def hasLeftChild(self, index):
        return self.getLeftChildIndex(index) < self.size
    def hasRightChild(self, index):
        return self.getRightChildIndex(index) < self.size
    def hasParent(self, index):
        return self.getParentIndex(index) >= 0

    def leftChild(self, index):
        return self.items[self.getLeftChildIndex(index)]
    def rightChild(self, index):
        return self.items[self.getRightChildIndex(index)]
    def parent(self, index):
        return self.items[self.getParentIndex(index)]

    
    def swap(self, a, b):
        self.items[a], self.items[b] = self.items[b], self.items[a]

    def ensureExtraCapacity(self):
        if self.size == self.capacity:
            self.items = self.items + [0 for _ in range(self.capacity)]
            self.capacity *= 2

    def peek(self):
        if self.size == 0:
            raise Exception("Heap Empty")
        return self.items[0]
    
    def poll(self):
        if self.size == 0:
            raise Exception("Heap Empty")
        item = self.items[0]
        self.items[0] = self.items[self.size-1]
        self.size -= 1
        self.heapifyDown()
        return item

    def add(self, item):
        self.ensureExtraCapacity()
        self.items[self.size] = item
        self.size += 1
        self.heapifyUp()

    def addList(self, nums):
        for i in nums:
            self.add(i)
        self.printHeap()

    def heapifyUp(self):
        index = self.size-1
        # while inside heap and parent is smaller than element added
        while self.hasParent(index) and self.parent(index) < self.items[index]:
            self.swap(self.getParentIndex(index), index)
            index = self.getParentIndex(index) 

    def heapifyDown(self):
        index = 0
        while self.hasLeftChild(index):
            largerChild = self.getLeftChildIndex(index)
            if self.hasRightChild(index) and self.rightChild(index) > self.leftChild(index):
                largerChild = self.getRightChildIndex(index)

            if self.items[index] > self.items[largerChild]:
                break
            else:
                self.swap(index, largerChild)

            index = largerChild

    def printHeap(self):
        for i in self.items:
            print(str(i) + "  ", end="")
        print("")

heap = MaxIntHeap(10)
nums = [5,3,17,10,84,19,6,22,9]
heap.addList(nums)
```

</details>

<details>
<summary>Trie [ Python ]</summary>

**Array-Based Trie**

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.doesWordEndHere = False
        self.children = [None for _ in range(100)]

class Trie:
    def __init__(self):
        self.root = Node("/")

    def getIndex(self, character):
        if character.isnumeric():
            return ord(character) - ord("0") + 52
        elif character.isupper():
            return ord(character) - ord("A") + 26
        elif character.islower():
            return ord(character) - ord("a")
        else:
            print(character)
            return ord(character) % 38 + 62

    def insert(self, node, word):
        index = self.getIndex(word[0])
        child = node.children[index]
        if child is None:
            child = Node(word[0])
            node.children[index] = child

        if len(word) == 1:
            if not child.doesWordEndHere:
                child.doesWordEndHere = True
                node.children[index] = child
                return True
            else:
                return False

        return self.insert(child, word[1:])

    def search(self, node, word):
        index = self.getIndex(word[0])
        child = node.children[index]
        if child is None:
            return False

        if len(word) == 1:
            if child.doesWordEndHere:
                return True
            else:
                return False

        return self.search(child, word[1:])

    def delete(self, node, word):
        index = self.getIndex(word[0])
        child = node.children[index]
        if child is None:
            return False

        if len(word) == 1:
            if child.doesWordEndHere:
                child.doesWordEndHere = False
                return True
            else:
                return False

        return self.delete(child, word[1:])

def multiStringSearch(bigString, smallStrings):
    trie = Trie()
    result = []
    words = bigString.split(" ")
    for i in words:
        # i = i.upper()
        for j in range(1, len(i)+1):
            # print(i)
            trie.insert(trie.root, i[:j])
    for i in smallStrings:
        # i = i.upper()
        # print(i)
        result.append(trie.search(trie.root, i))

    return result

bigString = "Is this particular test going to pass or is it going to fail? That is the question."
smallStrings = [
    "that",
    "the",
    "questions",
    "goes",
    "mountain",
    "passes",
    "passed",
    "going",
    "is"
]

print(multiStringSearch(bigString, smallStrings))
# print(trie.insert(trie.root, "abc"))
# print(trie.insert(trie.root, "abc"))
# print(trie.search(trie.root, "abc"))
# print(trie.delete(trie.root, "abc"))
# print(trie.search(trie.root, "abc"))
```

**Hashmap-Based Trie**

```python
class Node:
    def __init__(self, value):
        self.value = value
        self.doesWordEndHere = False
        self.children = {}

class Trie:
    def __init__(self):
        self.root = Node("/")

    def insert(self, node, word):
        child = node.children.get(word[0])
        if child is None:
            child = Node(word[0])
            node.children.update({word[0] : child})

        if len(word) == 1:
            if not child.doesWordEndHere:
                child.doesWordEndHere = True
                node.children.update({word[0] : child})
                return True
            else:
                return False

        return self
