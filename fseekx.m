function [res,position] = fseekx(offset,origin,position,re)

% global position;
%  m = memmapfile(fl,'Format',datatype) ;
%  global re;
% global posawal;
 
%   position 
%   re
  

% if(isempty(position))
% position = 0;
% end

if(origin == -1)
    position = offset;
end

if(origin == 0)
    position = position + offset;
end

if(origin == 1)
   
    position = re + offset;
end

if(position < 0 || position > re)


%if(position < 0 )
    res = -1;
%     res
%     pause
else
     res = 0;
end
% res
% position
% pause
return