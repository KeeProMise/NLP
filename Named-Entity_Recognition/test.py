import numpy as np

# Save
# dictionary = {'hello':'world',"i am":"zj"}
# dictionary2 = {"wo":"are"}
# lista = []
#
# lista.append(dictionary)
# lista.append(dictionary2)
#
# print(lista)
# np.save('my_file.npy', lista)
#
# # Load
# read_dictionary = np.load('my_file.npy',allow_pickle=True)
# print(read_dictionary)
# print(read_dictionary['hello'])
# print(read_dictionary['i am'])# displays "world"

file_handle=open('1.txt',mode='w',encoding='utf-8')

file_handle.write('hello word 你好 \n')
file_handle.writelines(['hello\n','world\n','你好\n','智游\n','郑州\n'])