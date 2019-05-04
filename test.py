
with open("gold_output.txt", "r") as f:
    content = f.readlines()

with open("output_normalized_feat.txt", "r") as f:
    content2 = f.readlines()

sum = corr = 0
for i, line in enumerate(content2):
    line_s = line[:-1].split(" - ")
    if line_s[2] == content[i][:-1].split(" - ")[1]:
        corr += 1
    sum += 1
print(sum,corr,corr*100.0/sum)