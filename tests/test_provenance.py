import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from provenance import git_blob_sha1


def test_git_blob_sha1_known_fixture(tmp_path):
    p=tmp_path/"x.txt"
    p.write_bytes(b"hello\n")
    assert git_blob_sha1(p) == "ce013625030ba8dba906f756967f9e9ca394464a"
