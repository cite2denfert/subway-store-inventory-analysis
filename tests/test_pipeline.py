"""합성 데이터로 전체 파이프라인을 실행하는 스모크 테스트.

실제 데이터(서울 열린데이터광장)와 같은 스키마를 흉내 낸다:
- '사용월' × '호선명' × '지하철역' 행, 시간대별 승차/하차 컬럼
- 환승역은 노선마다 행이 따로 있음
- 좌표 파일은 '역' 접미사가 붙고, 환승역이 노선별로 중복됨

    pytest -q
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import subway_regression_analysis as sra  # noqa: E402


def make_fake_data(tmp: Path, n_stations: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    names = [f"테스트{i}" for i in range(n_stations)] + ["서울역", "신촌(지하)"]
    transfer = set(names[:10])  # 2개 노선에 걸친 환승역
    rows = []
    for month in (202104, 202105, 202106):
        for name in names:
            size = rng.lognormal(10, 0.8)
            evening_bias = rng.normal(1, 0.25)
            for line in (["1호선", "2호선"] if name in transfer else ["1호선"]):
                row = {"사용월": month, "호선명": line, "지하철역": name}
                for h in sra.HOURS:
                    base = size * (0.02 + 0.06 * (h[:2] in {"07", "08", "18", "19"}))
                    board = base * (evening_bias if h[:2] in {"18", "19"} else 1)
                    row[f"{h} 승차인원"] = int(max(board * rng.uniform(0.8, 1.2), 1))
                    row[f"{h} 하차인원"] = int(max(base * rng.uniform(0.8, 1.2), 1))
                row["작업일자"] = 20210705
                rows.append(row)
    rid = pd.DataFrame(rows)
    rid_path = tmp / "ridership.csv"
    rid.to_csv(rid_path, index=False, encoding="cp949")

    loc_rows = []
    for name in names:
        label = name if name.endswith("역") or "(" in name else name + "역"
        for _ in range(2 if name in transfer else 1):
            loc_rows.append({"지하철역": label, "x좌표": 37.5, "y좌표": 127.0})
    loc_path = tmp / "loc.csv"
    pd.DataFrame(loc_rows).to_csv(loc_path, index=False, encoding="utf-8-sig")
    return rid_path, loc_path, len(names), transfer


def test_normalize_station():
    assert sra.normalize_station("서울역") == sra.normalize_station("서울")
    assert sra.normalize_station("신촌(지하)") == "신촌"
    assert sra.normalize_station(" 강남역 ") == "강남"
    assert sra.normalize_station("역촌") == "역촌"


def test_pipeline_end_to_end(tmp_path):
    rid, loc, n_names, transfer = make_fake_data(tmp_path)
    out = tmp_path / "out"
    res = sra.main(["--ridership", str(rid), "--locations", str(loc), "--out-dir", str(out)])

    # 환승역·좌표 중복이 있어도 역당 정확히 1행
    assert res["n_stations"] == n_names
    types = pd.read_csv(out / "station_types.csv", encoding="utf-8-sig")
    assert types["지하철역"].is_unique
    assert {"서울", "신촌"} <= set(types["지하철역"])

    # 환승역은 노선별 합산 (평균 아님): 2개 노선 역의 X는 1노선일 때의 약 2배 규모
    station = sra.load_ridership(rid, None)
    t = sra.normalize_station(sorted(transfer)[0])
    raw = pd.read_csv(rid, encoding="cp949")
    raw = raw[raw["지하철역"].map(sra.normalize_station) == t]
    expected = raw.groupby("사용월")[sra.MORNING_ALIGHT].sum().sum(axis=1).mean()
    assert np.isclose(station.loc[t, sra.MORNING_ALIGHT].sum(), expected)

    cl = res["classification"]
    assert cl["n_evening"] + cl["n_morning"] == n_names
    for f in ("fig1_regression_analysis.png", "fig2_hourly_pattern_heatmap.png",
              "fig3_top20_station_heatmap.png", "results.json"):
        assert (out / f).stat().st_size > 0
    json.loads((out / "results.json").read_text(encoding="utf-8"))
