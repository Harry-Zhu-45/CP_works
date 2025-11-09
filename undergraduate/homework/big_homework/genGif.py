import os
import sys

import imageio.v2 as imageio


def create_gif(source, name, duration):
    """png to gif

    input:
        source: png file list (sorted)
        name: name of the gif file
        duration: the time interval between each picture
    """
    frames = []
    for img in source:
        frames.append(imageio.imread(img))

    imageio.mimsave(name, frames, duration=duration)
    print("Convert complete!")


def main(or_path):
    """main function

    input:
        or_path: the path of the folder which contains the png files
    """
    path = os.chdir(or_path)
    pic_list = os.listdir()
    gif_name = "output.gif"  # the name of the gif file
    duration_time = 0.3
    create_gif(pic_list, gif_name, duration_time)


if __name__ == "__main__":
    param_list = sys.argv
    if len(param_list) != 2:
        print("请输入需要处理的文件夹！")
    else:
        main(param_list[1])
