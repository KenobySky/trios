import trios.shortcuts.persistence as p

if __name__ == '__main__':
    jung_op = p.load_gzip('trained-jung.op.gz')
    print(jung_op.cite_me())

