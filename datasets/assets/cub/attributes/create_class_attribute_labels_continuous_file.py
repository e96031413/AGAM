# 將CUB的class_attribute_labels_continuous從312個attribute縮減成156或更少，以證明AGAM的效果

"""
建立class_attribute_labels_continuous
"""


import random
random.seed(1)

total_class = 200

total_attribute = 312
attribute_to_delete = 156
remaining_attribute = total_attribute - attribute_to_delete

filename = 'class_attribute_labels_continuous.txt'

class_attribute_list=[]  # 200 class
each_attribute_list=[]

with open(filename) as file:
    for i in file:
        class_attribute_list.append(i.strip('\n').split(' '))

# print(len(class_attribute_list))     # 200 class
# print(len(class_attribute_list[0]))  # 312 attribute for each class

for i in range(0,total_class):
    for j in range(0,attribute_to_delete):
        class_attribute_list[i].remove(random.choice(class_attribute_list[i]))

# print(len(class_attribute_list))        # 200 class
# print(len(class_attribute_list[0]))     # delete half of the attributes, 156 remaining.

new_file_name = 'class_attribute_labels_continuous_{}.txt'.format(remaining_attribute)

with open(new_file_name, 'w') as file:
    for k in class_attribute_list:
        file.write(' '.join(k)+'\n')    
