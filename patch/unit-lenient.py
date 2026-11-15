"""Check the patched file without a GPU: print the acceptance function it will build under the current
environment. The module is not imported as a whole (TileLang does not load without a GPU); only the patch's two
insertions are executed. Run inside the image: python3 /unit.py"""
F = "/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py"
s = open(F).read()
ns = {}
i = s.index("import math as _night_math")
j = s.index("\n", s.index("_NIGHT_BY_TEMP ="))
exec(s[i:j + 1], ns)
k = s.index('_HEADER += (\n    "\\n// Lenient')
e = s.index("def _distances")
ns["_HEADER"] = ""
exec(s[k:e], ns)
h = ns["_HEADER"]
print(h[h.index("__device__"):].strip())
assert s.count('"vllm_night_lenient"') == 1, "the kernel calls the function once"
