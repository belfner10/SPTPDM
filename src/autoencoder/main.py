#                           0       1       2      3
# resolution 104424,161190 -2493045,2342655,177285,3310005

# top left -2493045 3310005
# bottom right 2342655 177285

# top left 280783 2189288
# bottom right 322268.1191856861 2149308.125005843
# xs yl xl ys
step = 300
m = step * 30
print((3310005 - 177285) / m)
print((2342655 + 2493045) / m)
print(104424*161190)
# print(280783.52000413515 2189288.9339491683 322268.1191856861 2149308.125005843)

for x in range(45,int((2342655 + 2493045) / m)):
    for y in range(300, int((3310005 - 177285) / m)):
        print(-2493045 + m * x, 177285 + m * y + m, -2493045 + m * x + m, 177285 + m * y)
        exit()
