from transcript_lok.ui import _progress_bar, _progress_status, _should_emit_progress


def test_progress_bar_status():
    assert _progress_bar(50) == "[##########----------]"
    status = _progress_status(5.0, 10.0, "cpu", "int8")
    assert "50%" in status
    assert "00:00:05.000 / 00:00:10.000" in status


def test_progress_updates_are_bucketed():
    should_emit, bucket, seconds = _should_emit_progress(1.0, 100.0, -1, -30.0)
    assert should_emit
    assert bucket == 0
    assert seconds == 1.0

    should_emit, bucket, seconds = _should_emit_progress(2.0, 100.0, bucket, seconds)
    assert not should_emit
    assert bucket == 0
    assert seconds == 1.0

    should_emit, bucket, seconds = _should_emit_progress(6.0, 100.0, bucket, seconds)
    assert should_emit
    assert bucket == 1
    assert seconds == 6.0
