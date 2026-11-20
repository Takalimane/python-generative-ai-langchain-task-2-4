from rag_agent_app.tools import calculate, read_local_file


def test_calculator():
    assert calculate("(12 + 3) * 2") == "30"


def test_calculator_rejects_code():
    assert "Calculator error" in calculate("__import__('os').system('echo bad')")


def test_local_file_reader_rejects_parent_traversal(tmp_path):
    assert "restricted" in read_local_file("../outside.txt")
