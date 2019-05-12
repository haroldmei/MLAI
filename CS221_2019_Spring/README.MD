
// The sculpture/room demo  
// Create your own factor graph!  
// Call variable(), factor(), query() followed by an inference algorithm.  
variable('X1', ['R', 'B'])  
variable('X2', ['R', 'B'])  
variable('X3', ['R', 'B'])  

factor('f1', 'X1', function(x1) {  
  return x1 == 'B';  
})  
factor('f2', 'X2', function(x2) {  
  return x2 == 'B';  
})  
factor('f3', 'X3', function(x3) {  
  return x3 == 'B';  
})  
factor('o1','X1 X2', function(x1,x2){  
  return x1 != x2;  
})  
factor('o2','X2 X3', function(x2,x3){  
  return x2 == x3;  
})  
factor('oo','X1 X2 X3', function(x1, x2, x3){  
  a1 = (x1 == 'B');  
  a2 = (x2 == 'B');  
  a3 = (x3 == 'B');  
  return (a1+a2+a3)<=1;  
})  
sumVariableElimination()  

    
    
// Evemt Scheduling  
// The sculpture/room demo  
// Create your own factor graph!  
// Call variable(), factor(), query() followed by an inference algorithm.  
variable('A', [0, 1, 2, 3])  
variable('B', [0, 1, 2, 3])  
variable('C', [0, 1, 2, 3])  

factor('e1', 'A', function(a) {  
  return a <= 1;  
})  
factor('e2', 'B', function(b) {  
  return b == 1;  
})  
factor('e3', 'C', function(c) {  
  return c >= 2;  
})  
factor('f1', 'A B', function(a, b) {  
  return a != b;  
})  
factor('f2', 'B C', function(b, c) {  
  return b != c;  
})  
factor('f3', 'A C', function(a, c) {  
  return a != c;  
})  

sumVariableElimination()  
 
 
 
### bulbs and switches
// Create your own factor graph!  
// Call variable(), factor(), query() followed by an inference algorithm.  

variable('S1', [0, 1])  
variable('S2', [0, 1])  
variable('S3', [0, 1])  

factor('b1', 'S1 S2', function(a, b) {  
  return a || b;  
})  
factor('b2', 'S2 S3', function(a, b) {  
  return a || b;  
})  
sumVariableElimination()  
