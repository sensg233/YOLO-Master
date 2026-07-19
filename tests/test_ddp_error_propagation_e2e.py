import subprocess
import sys


MARKER = "DDP_E2E_UNIQUE_MARKER_7F3A"


def test_torchrun_rank1_marker_reaches_parent_output(tmp_path):
    worker = tmp_path / "worker.py"
    logs = tmp_path / "logs"
    worker.write_text(
        "import os\n"
        "from torch.distributed.elastic.multiprocessing.errors import record\n"
        "@record\n"
        "def main():\n"
        "    if int(os.environ['LOCAL_RANK']) == 1:\n"
        f"        raise RuntimeError('{MARKER}')\n"
        "if __name__ == '__main__':\n"
        "    main()\n",
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--master_addr=127.0.0.1",
        "--master_port=29683",
        "--nproc_per_node=2",
        "--log-dir",
        str(logs),
        "--tee",
        "3",
        str(worker),
    ]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=45)
    assert result.returncode != 0
    assert MARKER in result.stdout
    assert "Root Cause" in result.stdout
    assert list(logs.rglob("error.json"))
