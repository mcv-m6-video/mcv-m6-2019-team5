import argparse


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('methods', help='Method list separated by ;')

    args = parser.parse_args()

    method_refs = {
    }


if __name__ == '__main__':
    main()
