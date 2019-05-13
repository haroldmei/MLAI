import heapq


def minMoves(maze, x, y):
    # Write your code here
    w = len(maze[0])
    l = len(maze)
    visited = []
    for i in range(l):
        visited.append([0] * w)

    h = []
    heapq.heappush(h, (0,[(0,0)]))
    i = 0
    j = 0
    visited[0][0] = 1
    while i != x or j != y:
        cur = heapq.heappop(h)
        l = len(cur[1])
        n = (cur[0] + 1,[])
        for j in range(l):
            p1 = cur[1][j][0]
            p2 = cur[1][j][1]

            if p1 < w-1 and maze[p1+1][p2] != 1 and visited[p1+1][p2] == 0:#right
                visited[p1+1][p2] = 1
                n[1].append((p1+1, p2))
            if p1 > 0 and maze[p1-1][p2] != 1 and visited[p1-1][p2] == 0:#left
                visited[p1-1][p2] = 1
                n[1].append((p1-1, p2))
            if p2 < w-1 and maze[p1][p2+1] != 1 and visited[p1][p2+1] == 0:#right
                visited[p1][p2+1] = 1
                n[1].append((p1, p2+1))
            if p2 > 0 and maze[p1][p2-1] != 1 and visited[p1][p2-1] == 0:#left
                visited[p1][p2-1] = 1
                n[1].append((p1, p2-1))

        heapq.heappush(h, n)
    cur = heapq.heappop(h)
    return cur[0] + 1
    
mz = [[0,2,0],[0,0,1],[1,1,1]]
x=1
y=1
minMoves(mz,x,y)