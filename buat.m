tic

pfile = '/home/lei/Documents/Data/P40448.7';
 m = memmapfile(pfile)
 data = m.Data;


ax = 158252
for i=1:32
    
  aa0(i) = ax + ((90*976)+88816*9)*(i-1)+1;
  %aa1(i) = aa0+1
  aa2(i) = aa0(i)+(90*976)-1;
  ax = aa0(i)-1;
  uu(i,:)=aa0(i):aa2(i);
  %size(dt)
end

dat = data(uu);

toc