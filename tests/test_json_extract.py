from app.utils.json_extract import extract_json_object


def test_plain_object():
    assert extract_json_object('{"ok": true}') == {"ok": True}


def test_fenced_object():
    assert extract_json_object('```json\n{"ok": true}\n```') == {"ok": True}


def test_prose_around_object():
    assert extract_json_object('here you go\n{"ok": true}\nthanks') == {"ok": True}
