import fire
from .cli import morefish_cli

if __name__ == '__main__':
    main()


def main():
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(morefish_cli)