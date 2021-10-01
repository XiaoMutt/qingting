import math
import operator
import typing as tp


class SegmentTree(object):
    def __init__(self, request_capacity, operation, neutral_element):
        """
        Build a Segment Tree data structure.
        Adopted from:
        https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

        The Segment Tree is a binary search tree with a fixed capacity. It stores values like an array.
        The values can be set or get at index [0, capacity)

        However, there are additional structures that the tree maintains to allow fast segment-wise queries:
        The Segment Tree nodes storing the value of an given "operation" of both left and right children.
        For example the operation can be sum. In this case, a node stores the sum of both its children and represent
        the sum of it's children's segment range.

        To build the Segment Tree, the capacity must be a power of 2 (for building a completely balanced binary tree):
              1
             22
            3333
          44444444 <- raw array data

        The tree is fattened in a tree array like this:
            0122333344444444
        Therefore, the tree use a array of size 2 * capacity as storage:
            - the 0 element is not used, the 1 element is the tree root
            - the branch nodes at the first half of the tree array stores the segment "operation" values
            - the leaf nodes at the second half of the tree array stores the raw values

        Index mapping:
            - The raw array index `i` is stored in tree array index `capacity + i` ( the bottom leaf layer)
            - The tree node `idx` has a parent node at `idx//2` in tree array
            - The tree node `idx` has a left child at `idx*2` in tree array
            - The tree node `idx` has a right child at `idx*2 + 1` in tree array

        :param request_capacity: the requested capacity
        :param operation: the aggregation operation to perform on child nodes. E.g. sum, min, max, etc
        :param neutral_element: neutral element to fill the tree array for the operation.
            E.g. 0 for sum, float('-inf') for max, float('inf') for min
        """
        self._capacity = 1 << math.ceil(math.log2(request_capacity))
        self._requested_capacity = request_capacity
        self._tree_array = [neutral_element for _ in range(2 * self._capacity)]
        self._operation = operation

    def _reduce_helper(self, query_start, query_end, node, node_start, node_end):
        """
        Query the [query_start, query_end] region against the node whose value representing the segment of
        [node_start, node_end]
        :param query_start: query_start index
        :param query_end: query_end index
        :param node: the current node
        :param node_start: the start index of the current node's segment
        :param node_end: # the end index of the current node's segment
        :return: the queried value
        """
        # binary search for queried [query_start, query_end] against the [node_start, node_end] segment
        if query_start == node_start and query_end == node_end:
            # search converged
            return self._tree_array[node]
        mid = (node_start + node_end) // 2
        if query_end <= mid:
            # [query_start, query_end] falls in left segment
            return self._reduce_helper(query_start, query_end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= query_start:
                # [query_start, query_end] falls in right segment
                return self._reduce_helper(query_start, query_end, 2 * node + 1, mid + 1, node_end)
            else:
                # [query_start, query_end] split by mid
                return self._operation(
                    self._reduce_helper(query_start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, query_end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start: int = 0, end: tp.Optional[int] = None):
        """
        Returns result of applying `self.operation` to a contiguous subsequence of the array on the segment
        [start, end]. Notice: end is inclusive
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        :param start: beginning of the subsequence
        :param end: int end of the subsequences
        :return result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._requested_capacity - 1
        if end < 0:
            end += self._requested_capacity
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._tree_array[idx] = val
        idx //= 2  # parent
        while idx >= 1:  # not exceeding root
            self._tree_array[idx] = self._operation(
                self._tree_array[2 * idx],  # left child
                self._tree_array[2 * idx + 1]  # right child
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._tree_array[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            request_capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def max_idx(self, ceiling: float):
        """
        Find the maximum index `i` in the raw array such that
            sum(arr[0] + arr[1] + ... + arr[i]) <= ceiling
        ATTENTION: this only works for arrays containing non-negative numbers

        If the raw array values are discrete probabilities, then the accumulated sum(arr[0] + arr[1] + ... + arr[i])
        is the CDF(i). This function can be used to sample indices according to the probabilities using inverse CDF:
            - sample a probability `p` uniformly in [0,1]
            - find the maximum index `i` that sum(arr[0] + arr[1] + ... + arr[i]) <= p
            - then the `i`s will follow the distribution defined by the raw array.

        ATTENTION: will return -1 if no element satisfy the constrain
        :param ceiling: the target ceiling sum
        :return the maximum index satisfying the sum constraint sum(arr[0] + arr[1] + ... + arr[i]) <= ceiling, if
        exists; otherwise return -1.
        """

        idx = 1  # root
        while idx < self._capacity:  # while non-leaf
            if self._tree_array[2 * idx] > ceiling:  # left child
                idx = 2 * idx  # goto left child
            else:
                # <= ceiling
                ceiling -= self._tree_array[2 * idx]  # calculate the remaining for right child
                idx = 2 * idx + 1  # goto right child.
        res = idx - self._capacity - int(self._tree_array[idx] > ceiling)
        return res


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            request_capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
