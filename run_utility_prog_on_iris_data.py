f = open('iris.data.txt');

for line in f.readlines():
  line = line.strip();
  if line == '':
    continue;

  a,b,c,d,e = line.split(',');

  # progA
  f = ''
  if float(a) < 5.0:
    f = 1;
  else:
    f = 0;

  # progB
  g = ''
  if float(a) + float(b) < 8.0:
    g = 1;
  else:
    g = 0;

  print '%s,%d,%d' % (line,f,g);
  
