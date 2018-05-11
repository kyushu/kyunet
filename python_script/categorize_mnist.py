
import pathlib
from shutil import copyfile

if __name__ == '__main__':




    mnist_train_label = '/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/training_label.txt';

    label_list = {}
    with open(mnist_train_label, 'r') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            # print("file: {}, label:{}".format(i+1, line))
            label_list[str(i+1)] = line



    img_dir = "/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/"
    dst_root = "/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/cat_train_images/"
    file_list_file = "/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/train_images.list"


    file_list = []
    with open(file_list_file, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            file_list.append(line)


    for line in file_list:
        src_file = img_dir + line
        name = line.split('/')[-1].split('.')[0]
        label = label_list[name]

        dst_dir = dst_root + '/' + label
        pathlib.Path(dst_dir).mkdir(parents=True, exist_ok=True)
        dst_file = dst_dir + '/' + name + '.png'
        copyfile(src_file, dst_file)
