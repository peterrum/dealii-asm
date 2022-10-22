
def main():
    configurations = []

    for k in ["1", "2", "v"]:
        for type in ["c", "l", "dg", "g-s-c", "g-s-n", "g-p-c", "g-p-n"]:
            for op in ["add", "none", "post", "pre", "symm"]:
                predicate = False
                if (k == "1" or k == "v") and type == "c" and (op == "post" or op == "pre" or op == "symm"):
                    predicate = True
                elif (type == "l" or type == "dg") and (op == "post" or op == "pre" or op == "symm"):
                    predicate = True
                elif (type == "g-s-c" or type == "g-p-c") and (op == "pre" or op == "symm"):
                    predicate = True
                elif type == "g-s-n" and (op == "none" or op == "post" or op == "pre" or op == "symm"):
                    predicate = True
                elif type == "g-p-n":
                    predicate = True

                if predicate:
                    configurations.append("%s-%s-%s" % (op, k, type))

    print(" ".join(configurations))

    print(len(configurations))

if __name__== "__main__":
  main()