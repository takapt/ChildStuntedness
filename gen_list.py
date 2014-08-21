import sys
for s in sys.stdin.readlines():
    print('"' + s.strip()+ '",')
