import glob
import re


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


if __name__ == "__main__":
    timestamp = 0
    with open('rgb.txt', 'w') as f:
        f.write("# color images \n")
        f.write("#file: care\n")
        f.write("# timestamp filename\n")
        for img in sorted(glob.glob("/home/wannes/storage/agent_1/rgb/GDumper_A/*.png"), key=numericalSort):
            name = img.split('/agent_1/')[1]
            line = str(timestamp) + " " + name
            f.write(line)
            f.write('\n')
            timestamp += 0.08
