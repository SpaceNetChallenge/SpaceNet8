# from https://github.com/Mitarushi/ACL-Python/blob/master/library/dsu.py
"""
MIT License

Copyright (c) 2020 Mitarushi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List


class DSU:
    def __init__(self, n: int) -> None:
        self._n = n
        self.parent_or_size = [-1] * n

    def merge(self, a: int, b: int) -> int:
        assert 0 <= a < self._n
        assert 0 <= b < self._n
        x, y = self.leader(a), self.leader(b)
        if x == y:
            return x
        if -self.parent_or_size[x] < -self.parent_or_size[y]:
            x, y = y, x
        self.parent_or_size[x] += self.parent_or_size[y]
        self.parent_or_size[y] = x
        return x

    def same(self, a: int, b: int) -> bool:
        assert 0 <= a < self._n
        assert 0 <= b < self._n
        return self.leader(a) == self.leader(b)

    def leader(self, a: int) -> int:
        assert 0 <= a < self._n
        stack = []
        while self.parent_or_size[a] >= 0:
            stack.append(a)
            a = self.parent_or_size[a]
        for i in stack:
            self.parent_or_size[i] = a
        return a

    def size(self, a: int) -> int:
        assert 0 <= a < self._n
        return -self.parent_or_size[self.leader(a)]

    def groups(self) -> List[List[int]]:
        leader_buf = [self.leader(i) for i in range(self._n)]
        group_size = [0] * self._n
        for i in leader_buf:
            group_size[i] += 1
        result = [[] for _ in range(self._n)]
        for i in range(self._n):
            result[leader_buf[i]].append(i)
        result = [i for i in result if i]
        return result
