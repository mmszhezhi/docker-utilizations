
import pandas as pd
import re
cpu = []
mem = []
i = 0
with open("data.txt") as f:
    lines = f.readlines()
    print(len(lines))
    for line in lines:
        print(i)
        i += 1
        if line.startswith("aa7e266aa269"):
            try:
                result = re.split(r'\s*',line)
                cpu.append(result[2].split("%")[0])
                mem.append(result[6].split("%")[0])
            except:
                pass

df = pd.DataFrame({'cpu':cpu,'mem':mem})
df.to_csv("data.csv")
