import random
random.seed(1)

total_class = 200
nums_of_image = 11788

total_attribute = 312
attribute_to_delete = 306   # 156, 262, 306
remaining_attribute = total_attribute - attribute_to_delete

filename = 'image_attribute_labels.txt'

class_attribute_list=[]  # 200 class
each_attribute_list=[]

k=[]
with open(filename, "r") as f:
    for line in f.read().split("\n"):
        k.append(line)

n_l=[]
i=0
for i in range(len(k)):
    if i % 312==0:
        n_l.append(k[0+i:i+312:])
        i+=312

for i in range(0,nums_of_image):
    for j in range(0,attribute_to_delete):
        n_l[i].remove(random.choice(n_l[i]))

new_file_name = 'image_attribute_labels_{}.txt'.format(remaining_attribute)

#　https://stackoverflow.com/a/13434166/13369757
with open(new_file_name, 'w') as file:
    for k in n_l:
        file.write("\n".join(map(str, k)))
