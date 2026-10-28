#!/usr/bin/env python3
"""Copy results.txt into README.md between the results markers."""
import re
readme = open("README.md").read()
res = open("results.txt").read().rstrip()
new = re.sub(r"(<!-- results:start -->\n).*?(\n<!-- results:end -->)",
             lambda m: m.group(1) + "```text\n" + res + "\n```" + m.group(2), readme, flags=re.S)
open("README.md", "w").write(new)
print("README updated")
