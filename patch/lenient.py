"""Near-tie acceptance for the DSpark draft in spark-ds41f 1107 (site experiment, 2026-11-15).

Exact verification accepts a draft token with probability p: the probability the target model gives that
token. This patch adds two rules. Both are off by default, and then the patched file verifies exactly as
the original does.

  Typical, level k: the draft token is always accepted when the target gives it at least
      k x (the probability of the target's best token); otherwise it is verified exactly, with probability p.
      Only the target's best token and near-ties are accepted; a clear draft miss gets no leniency.
  Lenience, level l: the draft token is accepted with probability min(1, p / l). This is the lenience of
      Leviathan et al. (arXiv 2211.17192) for a one-token draft. It raises acceptance for clear misses too;
      not recommended, kept for the record.

A rejected row is resampled as before (the target without the rejected token). Requests at temperature 0
are unchanged.

Environment:
  SPARK_NIGHT_TYPICAL=k           every sampled request uses the typical rule at level k (0 < k < 1)
  SPARK_NIGHT_LENIENCE=l          every sampled request uses lenience l (0 < l < 1); applied after typical
  SPARK_NIGHT_LENIENCE_BY_TEMP=1  the request names its own level through its temperature, which overrides
                                  the engine-wide levels:
                                    1 - k/100 (0.9905 < T < 0.9995; 0.997 means k = 0.3) for typical;
                                    1 + l/100 (1.0005 < T < 1.0095; 1.003 means l = 0.3) for lenience.
                                  Sampling itself runs at that temperature, so within 1% of 1.
                                  Any other temperature gets the unmodified log probability.

One vLLM file changes: v1/worker/gpu/spec_decode/tilelang_rejection.py (the TileLang acceptance kernel for the
greedy draft and block verification, the path production runs). The Triton path (probabilistic draft) is not
touched.

Usage: python3 -I lenient.py <directory holding tilelang_rejection.py taken from the image>.
It fails if the source in the image differs from the one this was written against.
"""
import pathlib
import sys

p = pathlib.Path(sys.argv[1]) / "tilelang_rejection.py"
src = p.read_text()

a = "_MAX_LANES = 32\n"
assert src.count(a) == 1, "constants anchor"
src = src.replace(a, a + '''
import math as _night_math
import os as _night_os


def _night_level(name: str) -> float:
    value = float(_night_os.environ.get(name, "1") or "1")
    return value if 0.0 < value < 1.0 else 1.0


# site patch (lenient): near-tie verification, see vllm_night_lenient in the header and lenient.py.
_NIGHT_LENIENCE = _night_level("SPARK_NIGHT_LENIENCE")
_NIGHT_TYPICAL = _night_level("SPARK_NIGHT_TYPICAL")
_NIGHT_BY_TEMP = _night_os.environ.get("SPARK_NIGHT_LENIENCE_BY_TEMP", "0") == "1"
''')

b = "def _distances(lanes: int) -> tuple[int, ...]:"
assert src.count(b) == 1, "header anchor"
src = src.replace(b, '''_HEADER += (
    "\\n// Lenient verification of a one-hot draft, from its log probability lp and its\\n"
    "// shifted logit (the log of its probability over the row's best). Typical level k:\\n"
    "// a draft within k of the best is accepted (log probability 0). Lenience l: lp plus\\n"
    "// -log(l), capped at 0, so the draft is accepted with min(1, p / l). A request whose\\n"
    "// temperature is 1 - k / 100 or 1 + l / 100 names its own level; the rest take the\\n"
    "// engine's. With no level set this returns lp.\\n"
    "__device__ __forceinline__ float vllm_night_lenient(float lp, float shifted, float temp) {\\n"
    + (
        "  const float d = temp - 1.0f;\\n"
        "  if (d > 0.0005f && d < 0.0095f) return vllm_log_ratio(lp, -logf(d * 100.0f));\\n"
        "  if (d < -0.0005f && d > -0.0095f) return shifted >= logf(-d * 100.0f) ? 0.0f : lp;\\n"
        if _NIGHT_BY_TEMP
        else ""
    )
    + (
        "  if (shifted >= %.9ef) return 0.0f;\\n" % _night_math.log(_NIGHT_TYPICAL)
        if _NIGHT_TYPICAL < 1.0
        else ""
    )
    + (
        "  return vllm_log_ratio(lp, %.9ef);\\n" % -_night_math.log(_NIGHT_LENIENCE)
        if _NIGHT_LENIENCE < 1.0
        else "  return lp;\\n"
    )
    + "}\\n"
)


''' + b)

c = '''                log_prob_s[step] = T.call_extern(
                    "float32",
                    "vllm_sub",
                    T.call_extern("float32", "vllm_shifted", logit, row_max[0]),
                    softplus,
                )
'''
assert src.count(c) == 1, "log probability anchor"
src = src.replace(c, '''                log_prob_s[step] = T.call_extern(
                    "float32",
                    "vllm_night_lenient",
                    T.call_extern(
                        "float32",
                        "vllm_sub",
                        T.call_extern("float32", "vllm_shifted", logit, row_max[0]),
                        softplus,
                    ),
                    T.call_extern("float32", "vllm_shifted", logit, row_max[0]),
                    temp,
                )
''')
p.write_text(src)
print("lenient: patched", p)
