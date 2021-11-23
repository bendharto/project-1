function [dr,nread,position] = freadx(readsize,datatype, datatot,position,re)
% global position;
% global re;

if(isequal(datatype,'int16'))
%     disp('ini int32')

% if(isempty(position))
% position = 0;
% end
readsize = readsize*2;
% m = memmapfile(fl,'Format',datatype) ;
% position
% %pause
if(position+readsize <= re)
% dr = res(position+1:position+readsize);
% dr=readsharedmem([position+0 readsize 5678 27]);
dr=datatot(position+1:position+readsize);
% if(~isequal(dr,drx))
%     pause
% end
%size(dr)
 %%pause
dr=typecast(uint8(dr), 'int16');
nread = readsize/2;
position = position+readsize;
% disp('ini yg satu')
% length(m.Data)


elseif(position+readsize > re)
    %disp('ini position+readsize > re')
    %%pause
    
    nread = floor((re - position)/2);
%     dr = res(position+1:re);
%     dr=readsharedmem([position+0 (nread*2) 5678 27]);
    dr=datatot(position+1:position+(nread*2));
%     if(~isequal(dr,drx))
%     pause
%     end
%     %%pause
    dr=typecast(uint8(dr), 'int16');
    position = position+(nread*2);
end
end



if(isequal(datatype,'int32'))
%     disp('ini int32')

% if(isempty(position))
% position = 0;
% end
readsize = readsize*4;
% m = memmapfile(fl,'Format',datatype) ;
if(position+readsize <= re)
% dr = res(position+1:position+readsize);
% dr=readsharedmem([position+0 readsize 5678 27]);
dr=datatot(position+1:position+readsize);
% if(~isequal(dr,drx))
%     pause
% end
% %pause
dr=typecast(uint8(dr), 'int32');
nread = readsize/4;
position = position+readsize;
% disp('ini yg satu')
% length(m.Data)



elseif(position+readsize > re)
    
   nread = floor((re - position)/4);
%     dr = res(position+1:re);
%    dr=readsharedmem([position+0 (nread*4) 5678 27]);
   dr=datatot(position+1:position+(nread*4));
%     if(~isequal(dr,drx))
%     pause
%     end
%     %pause
    dr=typecast(uint8(dr), 'int32');
    position = position+(nread*4);
end
end



return