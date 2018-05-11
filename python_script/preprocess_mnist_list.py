
import os


if __name__ == '__main__':

    mnist_train_list = '/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/train_images.list';
    mnist_train_label = '/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/training_label.txt';

    # Read image file
    file_list = []
    with open(mnist_train_list, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            file_list.append(line)
            # print("line: {}".format(line))

    # Read Label corresponding to image file
    label_list = {}
    with open(mnist_train_label, 'r') as f:
        for i, line in enumerate(f):
            line = line.rstrip('\n')
            # print("file: {}, label:{}".format(i+1, line))
            label_list[str(i+1)] = line

    mnist_train_image_label = '/mtdata/Dropbox/personal_proj/kyunet_proj/kyunet/example_data/mnist/training_file.txt'
    with open(mnist_train_image_label, 'w') as f:
        for file in file_list:
            file_name = file.split('/')[-1].split('.')[0]
            label = label_list[file_name]
            # print("file_name: {}: label: {}".format(file_name, label))
            write_line = file + ' ' + label + '\n'
            f.write(write_line)


