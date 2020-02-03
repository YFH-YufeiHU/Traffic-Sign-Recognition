import numpy as np
set = []
for i in range(5):
    set.append([i,i+1])
print(set)
result = np.array(set)
box_area =np.minimum([[5,1,6],[2,4,7]],[[2,2,2],[3,3,3]])
print(box_area)
