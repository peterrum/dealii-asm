import sys

def format(i):
    if i[1] == 0.0:
        return "& &"
    else:
        return "%.1f & %.1f & %.2f" % (i[1],i[2], i[0])


def main():
    op   = ["add", "none", "post", "pre", "symm"]
    type = ["1-c", "1-l", "1-dg", "1-g-s-c", "1-g-s-n", "1-g-p-c", "1-g-p-n", 
                   "2-l", "2-dg", "2-g-s-c", "2-g-s-n", "2-g-p-c", "2-g-p-n",
            "v-c", "v-l", "v-dg", "v-g-s-c", "v-g-s-n", "v-g-p-c", "v-g-p-n"]

    w = len(op)
    h = len(type)

    matrix = [[[0] * 3 for z in range(w)]  for y in range(h)] 

    myfile = open(sys.argv[1], 'r')
    data_t = myfile.readlines()
    myfile = open(sys.argv[2], 'r')
    data_r = myfile.readlines()
    myfile = open(sys.argv[3], 'r')
    data_w = myfile.readlines()

    for i in range(0, len(data_t)): 
        data        = data_t[i].strip().split(" ")
        labels      = data[1]
        labels      = labels.split("-")
        time        = float(data[4])
        dofs        = float(data[2])
        repetitions = float(data[3])

        data   = data_r[i].strip().split(",")
        read   = float(data[1])

        data   = data_w[i].strip().split(",")
        write  = float(data[1])

        label_c = labels[0]
        label_r = "-".join(["%s" % i for i in labels[1:]])

        index_r = type.index(label_r)
        index_c = op.index(label_c)

        matrix[index_r][index_c][0] = time
        matrix[index_r][index_c][1] = read  * 1e9 / dofs / repetitions / 4 # TODO
        matrix[index_r][index_c][2] = write * 1e9 / dofs / repetitions / 4 # TODO

    print("\\begin{tabular}{l|" + " | ".join(">{\\centering\\arraybackslash}p{0.7cm} >{\\centering\\arraybackslash}p{0.7cm} >{\\centering\\arraybackslash}p{1.0cm}" for i in op) + "}")
    print("\\toprule")

    print(" &" + " & ".join("\\multicolumn{3}{|c}{%s}" % i for i in op) + " \\\\")

    print("\midrule")

    print(" &" + " & ".join("$r$ & $w$ & $t[s]$" for i in op) + " \\\\")


    print("\\midrule")
    for r in range(0, h):
        if r == 7 or r == 13:
            print("\midrule")
        print(type[r] + " & " + " & ".join([format(i) for i in matrix[r]]) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

if __name__== "__main__":
  main()