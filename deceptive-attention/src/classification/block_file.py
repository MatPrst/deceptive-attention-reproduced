import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", dest="data_file", type=str)
    parser.add_argument('--block-words', dest='block_words', nargs='+', default=[],
        help = 'list of words you wish to block (default is None)')

    params = parser.parse_args()
    print(params)

    lines = []
    with open(params.data_file, "r") as f:
        for line in f.readlines():
            label, sentence = line.split("\t")
            lines.append(sentence.split())
    
    block_filename = f"{params.data_file}.block"
    with open(block_filename, "w") as f:
        block_lines = []
        for line in lines:
            block_line = [int(token in params.block_words) for token in line]
            print(" ".join(map(str, block_line)), file=f)
    

if __name__ == "__main__":
    main()