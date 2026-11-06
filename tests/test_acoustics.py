"""Real recordings: MIMII / DCASE 2020 Task 2 pump id_00, Opus-encoded like a phone voice note."""
from injini_floor import acoustics as A

S = "samples/pump-00-{}.ogg"


def clip(name):
    return A.analyse(open(S.format(name), "rb").read())


def test_pump_enrol_score_roundtrip():
    base = A.enrol("pump-00", [clip(f"healthy-{i}") for i in (1, 2, 3)])
    check, fault, after = (A.score(base, clip(n)) for n in ("check", "fault", "after-repair"))
    assert not check.anomalous and not after.anomalous
    assert fault.anomalous
    again = A.Baseline.from_json(base.to_json())
    assert abs(again.sigma(clip("fault").E) - fault.sigma) < 0.5
    print(f"\ncheck {check.sigma:+.1f}  fault {fault.sigma:+.1f}  after {after.sigma:+.1f}\n{fault.line()}")
