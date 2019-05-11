
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
