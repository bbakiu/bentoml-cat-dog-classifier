import os
import cv2

bad_list = []
dir = "PetImages"
subdir_list = os.listdir(
    dir
)  # create a list of the sub directories in the directory ie train or test
for d in subdir_list:  # iterate through the sub directories train and test
    dpath = os.path.join(dir, d)  # create path to sub directory
    if d in ["test", "train"]:
        class_list = os.listdir(dpath)  # list of classes ie dog or cat
        # print (class_list)
        for klass in class_list:  # iterate through the two classes
            class_path = os.path.join(dpath, klass)  # path to class directory
            # print(class_path)
            file_list = os.listdir(
                class_path
            )  # create list of files in class directory
            for f in file_list:  # iterate through the files
                fpath = os.path.join(class_path, f)
                index = f.rfind(".")  # find index of period infilename
                ext = f[index + 1 :]  # get the files extension
                if ext not in ["jpg", "png", "bmp", "gif"]:
                    print(f"file {fpath}  has an invalid extension {ext}")
                    bad_list.append(fpath)
                else:
                    try:
                        img = cv2.imread(fpath)
                        size = img.shape
                    except:
                        print(f"file {fpath} is not a valid image file ")
                        os.remove(fpath)
                        bad_list.append(fpath)

print("List of bad files\n")
print(bad_list)
