"""Step 1: reduce the yearly trip archives to a compact projection.

The archives are 2.8 GB of CSV each once decompressed, and there are two of them.
Three things were measured before this file was written, and all three shaped it:

  * Never extract. `unzip -p` streams the member; the decompressed file is never
    written to disk.
  * Never parse with Python's csv module -- it took 10+ minutes per season. awk
    does the same pass in well under a minute, so awk owns the streaming half and
    DuckDB owns the relational half.
  * Never aggregate in awk. The exploratory pass could, because a station-DAY
    grid is only ~470k keys; the hourly grid this model needs is ~8M keys, and a
    BSD awk associative array that size is a memory problem for no gain. So awk
    PROJECTS (one short line per trip) and DuckDB does the GROUP BY, which is
    what it is good at.

What comes out is `trips_<year>.tsv`: six integers per trip, station names
replaced by small ids with the dictionary written alongside. That is ~27 bytes a
row instead of ~90, and it makes the station dictionary a by-product rather than
a second pass.

Interned here rather than downstream: the local-hour conversion. It needs the DST
table (see paths.dst_intervals) and doing it once at projection time means every
later step is pure integer arithmetic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request

from . import paths

# Field positions in the 2024/2025 schema (verified byte-identical):
#   1 STARTSTATIONNAME  5 ENDSTATIONNAME  9 STARTTIMEMS  10 ENDTIMEMS
#
# About 0.4% of rows have an empty ENDSTATIONNAME and ENDTIMEMS -- trips that
# were never returned to a station. They are still real departures, so they are
# kept with an end marker of (0, -1, -1) and filtered out of the arrivals side
# downstream. Dropping the whole row would quietly bias departures.
AWK = r"""
function off(t,   i) {
  for (i = 0; i < nd; i++) if (t >= ds[i] && t < de[i]) return -14400
  return -18000
}
BEGIN {
  FS = ","; OFS = "\t"
  nd = split(dstart, ds_, ","); split(dend, de_, ",")
  for (i = 1; i <= nd; i++) { ds[i-1] = ds_[i] + 0; de[i-1] = de_[i] + 0 }
}
NR == 1 { next }
NF != 10 { bad++; next }
{
  st = $9 + 0
  if (st <= 0) { badtime++; next }
  ss = $1
  if (ss == "") { nostart++; next }
  if (!(ss in id)) { id[ss] = ++nid; name[nid] = ss }

  s = int(st / 1000); s += off(s)
  sd = int(s / 86400); sh = int((s % 86400) / 3600)

  es = $5; et = $10 + 0
  if (es != "" && et > 0) {
    if (!(es in id)) { id[es] = ++nid; name[nid] = es }
    e = int(et / 1000); e += off(e)
    print id[ss], sd, sh, id[es], int(e / 86400), int((e % 86400) / 3600)
  } else {
    incomplete++
    print id[ss], sd, sh, 0, -1, -1
  }
  n++
}
END {
  for (i = 1; i <= nid; i++) print i "\t" name[i] > dictfile
  printf "kept=%d stations=%d incomplete=%d malformed=%d empty_start=%d bad_time=%d\n",
         n, nid, incomplete, bad, nostart, badtime > "/dev/stderr"
}
"""


def download(year: int) -> paths.Path:
    zp = paths.DATA / f"{year}.zip"
    if zp.exists() and zp.stat().st_size > 1_000_000:
        return zp
    url = paths.ZIP_URLS[year]
    print(f"[opendata] downloading {year} from {url}", flush=True)
    tmp = zp.with_suffix(".zip.part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(zp)
    return zp


def project(year: int, force: bool = False) -> paths.Path:
    out = paths.DATA / f"trips_{year}.tsv"
    dictfile = paths.DATA / f"stations_{year}.tsv"
    if out.exists() and dictfile.exists() and not force:
        print(f"[opendata] {out.name} exists ({out.stat().st_size / 1e6:.0f} MB) -- skipping")
        return out

    zp = download(year)
    # The season can straddle a DST boundary, and a trip's start and end can too,
    # so every year that could appear in the file gets an interval.
    iv = paths.dst_intervals((year - 1, year, year + 1))
    t0 = time.time()
    print(f"[opendata] streaming {zp.name} -> {out.name}", flush=True)

    unzip = subprocess.Popen(["unzip", "-p", str(zp)], stdout=subprocess.PIPE)
    with open(out, "wb") as fh:
        awk = subprocess.Popen(
            [
                "awk",
                "-v", f"dstart={','.join(str(a) for a, _ in iv)}",
                "-v", f"dend={','.join(str(b) for _, b in iv)}",
                "-v", f"dictfile={dictfile}",
                AWK,
            ],
            stdin=unzip.stdout,
            stdout=fh,
        )
        unzip.stdout.close()  # let unzip see SIGPIPE if awk dies
        rc = awk.wait()
    unzip.wait()
    if rc != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"awk failed on {year} (exit {rc})")
    print(f"[opendata] {year} done in {time.time() - t0:.0f}s, {out.stat().st_size / 1e6:.0f} MB", flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", type=int, nargs="+", default=list(paths.TRAIN_YEARS))
    ap.add_argument("--force", action="store_true", help="re-project even if the output exists")
    a = ap.parse_args(argv)
    paths.ensure_dirs()
    for y in a.years:
        project(y, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
