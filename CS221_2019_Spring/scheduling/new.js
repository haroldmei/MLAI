// bulbs and switchs
// Create your own factor graph!
// Call variable(), factor(), query() followed by an inference algorithm.
variable('S1', [0, 1])
variable('S2', [0, 1])
variable('S3', [0, 1])
variable('S4', [0, 1])
variable('S5', [0, 1])

factor('b1', 'S1 S2', function(a, b) {
return a ^ b;
})
factor('b2', 'S2 S3 S4', function(a, b, c) {
return a ^ b ^ c;
})
factor('b3', 'S3 S4 S5', function(a, b, c) {
return a ^ b ^ c;
})
sumVariableElimination()



def enforceAC3(x_i, x_j):
d_i = self.domains[x_i]
d_j = self.domains[x_j]
changed = False
for i in list(d_i):
    remove = True
    for j in list(d_j):
        if self.csp.binaryFactors[x_j][x_i][j]:
            if self.csp.binaryFactors[x_j][x_i][j][i] != 0:
                remove = False
    if remove:
        changed = True
        self.domains[x_i].remove(i)
return changed

q = [var]
while q:
var1 = q.pop(0)
nbrs = self.csp.get_neighbor_vars(var1)
for var2 in nbrs:
    changed = enforceAC3(var2,var1)
    if changed:
        q.append(var2)

