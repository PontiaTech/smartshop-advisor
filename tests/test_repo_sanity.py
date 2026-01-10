import pathlib

def test_main_file_exists():
    main_file = pathlib.Path("app/main.py")
    assert main_file.exists(), "app/main.py no existe"

def test_main_is_python_file():
    content = pathlib.Path("app/main.py").read_text(encoding="utf-8")
    assert "FastAPI" in content
    assert "app = FastAPI" in content
