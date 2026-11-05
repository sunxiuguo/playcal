import json
import random
import struct
import subprocess
import sys
import zlib

import pytest

from playcal.visual import PngError, SIG, diff, read_png, write_png


def encode(path, w, h, rows, ctype, bpp, depth=8, interlace=0):
    """Test-side PNG writer that uses every filter type, to exercise the reader's unfiltering."""
    chunk = lambda k, b: struct.pack(">I", len(b)) + k + b + struct.pack(">I", zlib.crc32(k + b))
    passes = [rows]
    if interlace == 1:
        # PNG's repeating 8x8 pass map, independent of the reader's pass geometry.
        grid = ("16462646", "77777777", "56565656", "77777777",
                "36463646", "77777777", "56565656", "77777777")
        passes = []
        for number in "1234567":
            selected = [b"".join(row[x * bpp:(x + 1) * bpp] for x in range(w)
                                 if grid[y % 8][x % 8] == number) for y, row in enumerate(rows)]
            passes.append([row for row in selected if row])
    raw = bytearray()
    for number, pass_rows in enumerate(passes):
        prev = bytes(len(pass_rows[0])) if pass_rows else b""
        for y, row in enumerate(pass_rows):
            f = (y + number) % 5
            out = bytearray()
            for i, v in enumerate(row):
                a = row[i - bpp] if i >= bpp else 0
                b, c = prev[i], (prev[i - bpp] if i >= bpp else 0)
                if f == 4:
                    p = a + b - c
                    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                    pred = a if pa <= pb and pa <= pc else b if pb <= pc else c
                else:
                    pred = [0, a, b, (a + b) // 2][f]
                out.append((v - pred) & 255)
            raw += bytes([f]) + out
            prev = row
    path.write_bytes(SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, interlace))
                     + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b""))


def test_reader_undoes_every_filter(tmp_path):
    rnd = random.Random(1)
    w, h = 7, 10
    rgb = [bytes(rnd.randrange(256) for _ in range(w * 3)) for _ in range(h)]
    encode(tmp_path / "a.png", w, h, rgb, 2, 3)
    _, _, rows = read_png(tmp_path / "a.png")
    assert [r[0::4] + r[1::4] for r in rows] == [r[0::3] + r[1::3] for r in rgb]
    assert all(r[3::4] == b"\xff" * w for r in rows)
    write_png(tmp_path / "b.png", w, h, rows)
    assert read_png(tmp_path / "b.png")[2] == rows


def test_diff_counts_only_pixels_over_tolerance():
    base = (4, 2, [b"\x10\x10\x10\xff" * 4, b"\x10\x10\x10\xff" * 4])
    new = (4, 2, [b"\x14\x10\x10\xff" * 4, b"\x10\x10\x10\xff" * 3 + b"\xff\x10\x10\xff"])
    out, rows = diff(base, new, tolerance=8)
    assert (out["changed_pixels"], out["box"]) == (1, [3, 1, 3, 1])
    assert rows[1][12:16] == b"\xff\x00\x00\xff"
    assert diff(base, (2, 2, base[2]))[0]["same_size"] is False
    assert diff(base, new, tolerance=8, ignore=[(3, 0, 3, 1)])[0]["changed_pixels"] == 0


def test_cli_save_then_check(tmp_path):
    w, h = 10, 10
    rows = [b"\x20\x40\x60\xff" * w for _ in range(h)]
    write_png(tmp_path / "shot.png", w, h, rows)
    run = lambda *a: subprocess.run([sys.executable, "-m", "playcal", "visual", *a, "--dir", str(tmp_path / "b"), "--json"],
                                    capture_output=True, text=True)
    assert run("save", "lvl", str(tmp_path / "shot.png")).returncode == 0
    r = run("check", "lvl", str(tmp_path / "shot.png"))
    assert r.returncode == 0 and json.loads(r.stdout)["verdict"] == "ok"
    rows[5] = b"\x20\x40\x60\xff" * 5 + b"\xff\xff\xff\xff" * 5
    write_png(tmp_path / "shot2.png", w, h, rows)
    r = run("check", "lvl", str(tmp_path / "shot2.png"))
    out = json.loads(r.stdout)
    assert r.returncode == 1 and out["changed_pixels"] == 5 and (tmp_path / "b" / "lvl.diff.png").exists()
    assert run("check", "lvl", str(tmp_path / "shot2.png"), "--ignore", "5,5,9,5").returncode == 0
    assert run("check", "lvl", str(tmp_path / "shot2.png"), "--ignore", "5,5").returncode == 2


def test_16_bit_png_is_read_at_8_bits(tmp_path):
    rnd = random.Random(2)
    w, h = 5, 6
    # RGBA, 16 bits per sample, big-endian: the high byte must survive, the low byte is dropped.
    rgba16 = [bytes(rnd.randrange(256) for _ in range(w * 8)) for _ in range(h)]
    encode(tmp_path / "deep.png", w, h, rgba16, 6, 8, depth=16)
    got_w, got_h, rows = read_png(tmp_path / "deep.png")
    assert (got_w, got_h) == (w, h)
    assert rows == [r[0::2] for r in rgba16]
    # 16-bit grey, no alpha, every filter type again
    grey16 = [bytes(rnd.randrange(256) for _ in range(w * 2)) for _ in range(h)]
    encode(tmp_path / "grey.png", w, h, grey16, 0, 2, depth=16)
    rows = read_png(tmp_path / "grey.png")[2]
    assert [r[0::4] for r in rows] == [r[0::2] for r in grey16] and all(r[3::4] == b"\xff" * w for r in rows)


@pytest.mark.parametrize("w,h", [(1, 1), (1, 9), (9, 1), (2, 2), (3, 5), (5, 3),
                                  (7, 9), (8, 8), (9, 7), (17, 33)])
@pytest.mark.parametrize("ctype,channels", [(0, 1), (2, 3), (4, 2), (6, 4)])
@pytest.mark.parametrize("depth", [8, 16])
def test_adam7_matches_plain_png(tmp_path, w, h, ctype, channels, depth):
    rnd = random.Random(3)
    bpp = channels * (depth // 8)
    raw_rows = [bytes(rnd.randrange(256) for _ in range(w * bpp)) for _ in range(h)]
    expected = []
    for row in raw_rows:
        rgba = bytearray()
        for x in range(w):
            samples = row[x * bpp:(x + 1) * bpp:depth // 8]
            rgb = samples[:1] * 3 if ctype in (0, 4) else samples[:3]
            rgba += rgb + (samples[-1:] if ctype in (4, 6) else b"\xff")
        expected.append(bytes(rgba))
    plain, interlaced = tmp_path / "plain.png", tmp_path / "adam7.png"
    encode(plain, w, h, raw_rows, ctype, bpp, depth)
    encode(interlaced, w, h, raw_rows, ctype, bpp, depth, interlace=1)
    assert read_png(plain) == (w, h, expected)
    assert read_png(interlaced) == (w, h, expected)
    assert diff(read_png(plain), read_png(interlaced), tolerance=0)[0]["changed_pixels"] == 0


@pytest.mark.parametrize("baseline_interlace", [0, 1])
def test_cli_checks_adam7_against_plain_png(tmp_path, baseline_interlace):
    w, h = 9, 9
    rows = [b"\x20\x40\x60\xff" * w for _ in range(h)]
    baseline, shot = tmp_path / "baseline.png", tmp_path / "shot.png"
    encode(baseline, w, h, rows, 6, 4, interlace=baseline_interlace)
    encode(shot, w, h, rows, 6, 4, interlace=1 - baseline_interlace)
    run = lambda *a: subprocess.run([sys.executable, "-m", "playcal", "visual", *a,
                                    "--dir", str(tmp_path / "b"), "--json"], capture_output=True, text=True)
    assert run("save", "lvl", str(baseline)).returncode == 0
    same = run("check", "lvl", str(shot), "--tolerance", "0")
    assert same.returncode == 0 and json.loads(same.stdout)["changed_pixels"] == 0
    rows[8] = rows[8][:-4] + b"\xff\xff\xff\xff"
    encode(shot, w, h, rows, 6, 4, interlace=1 - baseline_interlace)
    changed = run("check", "lvl", str(shot), "--tolerance", "0", "--max-changed", "0")
    result = json.loads(changed.stdout)
    assert changed.returncode == 1 and result["changed_pixels"] == 1 and result["box"] == [8, 8, 8, 8]
    diff_rows = read_png(tmp_path / "b" / "lvl.diff.png")[2]
    assert diff_rows[8][-4:] == b"\xff\x00\x00\xff"


def test_reader_rejects_unknown_interlace_method(tmp_path):
    path = tmp_path / "invalid.png"
    encode(path, 1, 1, [b"\x00\x00\x00\xff"], 6, 4, interlace=2)
    with pytest.raises(PngError, match="interlace"):
        read_png(path)
