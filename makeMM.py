import arena_util as au
import sys
argv = sys.argv[1]
filename = argv.split("/")[-1]
filepath = "/".join(argv.split("/")[:-1])
row = 0
N = 0
jsonfile = au.load_json(argv)
with open("/".join(sys.argv[1].split("/")[:-1])  + "/" + filename[:-5] + "_mm.txt", "w") as f:
    for query in jsonfile:
        N += len(query["songs"])
        row += 1
    f.write("%d %d %d\n" % (int(row), 707989, int(N)))
    row = 0
    for query in jsonfile:
        for song in query["songs"]:
            f.write("%d %d %d\n" % (int(row), int(song), 1))
        row += 1
f.close()
