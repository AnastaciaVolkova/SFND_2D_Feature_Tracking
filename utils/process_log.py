import pickle
import pandas as pd
import numpy as np


def key_points_info(data):
    data_x = dict()
    detectors = list()
    for record in data:
        if record[1] is not None:
            det, des = record[0].split("_")
            if det not in detectors:
                for metric, column in record[1]["detector"].items():
                    data_x[(det, metric)] = column
    data_det = pd.DataFrame.from_dict(data_x)
    data_det.to_csv("detectors.csv", index=False)


def matchers_info(data):
    data_x = list()
    for record in data:
        if record[1] is not None:
            det, des = record[0].split("_")
            for metric, column in record[1]["matcher"].items():
                data_x.append(((det, des, metric), column))
    data_x.sort(key=lambda x: x[0][0])
    data_x = dict((x, y) for x, y in data_x)
    data_det = pd.DataFrame.from_dict(data_x)
    data_det.to_csv("matchers.csv", index=False)


def det_des_time_info(data):
    data_x = dict()
    for record in data:
        if record[1] is not None:
            det, des = record[0].split("_")
            data_x[(record[0], f"detector {det}, ms")] = record[1]["detector"]["t"]
            data_x[(record[0], f"descriptor {des}, ms")] = record[1]["descriptor"]["t"]

    data_tms = pd.DataFrame.from_dict(data_x)
    print(data_tms)
    data_tms.to_csv("timing.csv", index=False)

    det_des_timing = list()
    data_tms_cols = list(set(x[0] for x in data_tms.columns))
    for det_des in data_tms_cols:
        mx = data_tms[det_des].to_numpy()
        det_des_timing.append((det_des, np.mean(mx.sum(axis=1))))
    det_des_timing.sort(key=lambda x: x[1])
    for comb, t in det_des_timing:
        print(f"{comb},{t}")


def main():
    with open("logs.pk", "rb") as fid:
        data = pickle.load(fid)
    key_points_info(data)
    matchers_info(data)
    det_des_time_info(data)


if __name__ == "__main__":
    main()
