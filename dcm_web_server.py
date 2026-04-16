"""
DICOM Phantom Generator – Web Server
=====================================
Flask backend that serves the HTML UI and processes DICOM generation
requests using the core functions from custom_dcm_gen.py.
"""

import os
import sys
import json
import threading
import datetime
import shutil

from flask import Flask, render_template, request, jsonify

# Import the core generation functions from custom_dcm_gen
from custom_dcm_gen import (
    write_dicom,
    generate_ct_slice,
    generate_mr_slice,
    _PLANE_META,
    PIXEL_GENERATORS,
)
from pydicom.uid import generate_uid
import random

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# Global state for tracking generation progress
generation_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "done": False,
    "error": None,
    "output_dir": "",
    "log": [],
}


def reset_status():
    generation_status.update({
        "running": False,
        "progress": 0,
        "total": 0,
        "message": "",
        "done": False,
        "error": None,
        "output_dir": "",
        "log": [],
    })


def generate_dicoms_thread(config):
    """
    Background thread that generates DICOM files based on user configuration.

    Config keys:
        modality        : "CT" or "MR"
        num_studies     : int
        num_series      : int  (per study)
        patient_id_mode : "same" or "different"
        study_uid_mode  : "same" or "different"
        date_mode       : "same" or "different"
        slices_per_series : int (default 12)
    """
    try:
        modality          = config["modality"]
        num_studies       = int(config["num_studies"])
        num_series        = int(config["num_series"])
        patient_id_mode   = config["patient_id_mode"]
        study_uid_mode    = config["study_uid_mode"]
        date_mode         = config["date_mode"]
        slices_per_series = int(config.get("slices_per_series", 12))
        image_size        = 512
        slice_thickness   = 5.0

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(__file__),
            f"generated_dicoms_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)
        generation_status["output_dir"] = output_dir

        total_series_count = num_studies * num_series
        generation_status["total"] = total_series_count * slices_per_series
        generation_status["progress"] = 0

        pm  = _PLANE_META[modality]
        gen = PIXEL_GENERATORS[modality]

        # ── Generate Patient IDs ──────────────────────────────────────────
        if patient_id_mode == "same":
            patient_ids   = ["PAT-100001"] * num_studies
            patient_names = ["PHANTOM^PATIENT"] * num_studies
        else:
            patient_ids   = [f"PAT-{100001 + i}" for i in range(num_studies)]
            patient_names = [f"PHANTOM^PATIENT_{i + 1}" for i in range(num_studies)]

        # ── Generate Study Instance UIDs ──────────────────────────────────
        if study_uid_mode == "same":
            shared_study_uid = generate_uid()
            study_uids = [shared_study_uid] * num_studies
        else:
            study_uids = [generate_uid() for _ in range(num_studies)]

        # ── Generate Study Dates ──────────────────────────────────────────
        base_date = datetime.date(2026, 4, 1)
        if date_mode == "same":
            study_dates = [base_date.strftime("%Y%m%d")] * num_studies
        else:
            study_dates = [
                (base_date + datetime.timedelta(days=i * 2)).strftime("%Y%m%d")
                for i in range(num_studies)
            ]

        study_time = "090000"

        # ── Generate Series Instance UIDs (always unique per series slot) ─
        series_uids_per_study = []
        for _ in range(num_studies):
            series_uids_per_study.append(
                [generate_uid() for _ in range(num_series)]
            )

        log_msg = (
            f"Configuration:\n"
            f"  Modality       : {modality}\n"
            f"  Studies        : {num_studies}\n"
            f"  Series/study   : {num_series}\n"
            f"  Patient ID     : {patient_id_mode}\n"
            f"  Study UID      : {study_uid_mode}\n"
            f"  Date           : {date_mode}\n"
            f"  Slices/series  : {slices_per_series}\n"
            f"  Output         : {output_dir}\n"
        )
        generation_status["log"].append(log_msg)
        generation_status["message"] = "Starting generation..."

        global_series_idx = 0

        for study_idx in range(num_studies):
            pid   = patient_ids[study_idx]
            pname = patient_names[study_idx]
            suid  = study_uids[study_idx]
            sdate = study_dates[study_idx]
            date_fmt = f"{sdate[:4]}-{sdate[4:6]}-{sdate[6:]}"

            study_msg = (
                f"Study {study_idx + 1}/{num_studies}  |  "
                f"Patient: {pid}  |  Date: {date_fmt}  |  "
                f"StudyUID: ...{suid[-12:]}"
            )
            generation_status["log"].append(study_msg)

            for ser_idx in range(num_series):
                global_series_idx += 1
                series_uid = series_uids_per_study[study_idx][ser_idx]
                series_num = random.randint(100, 999)

                folder_name = (
                    f"Study{study_idx + 1}_Series{ser_idx + 1:02d}"
                    f"_{modality}_{pm['plane']}_{date_fmt}_S{series_num}"
                )
                folder = os.path.join(output_dir, folder_name)
                os.makedirs(folder, exist_ok=True)

                series_msg = (
                    f"  Series {ser_idx + 1}/{num_series}  |  "
                    f"SeriesUID: ...{series_uid[-12:]}  |  "
                    f"{slices_per_series} slices"
                )
                generation_status["log"].append(series_msg)
                generation_status["message"] = (
                    f"Study {study_idx + 1}/{num_studies}, "
                    f"Series {ser_idx + 1}/{num_series} ..."
                )

                for sl in range(slices_per_series):
                    filepath = os.path.join(folder, f"slice_{sl + 1:04d}.dcm")
                    slice_loc = (
                        round(sl * slice_thickness, 3)
                        if modality == "CT"
                        else round(
                            -(slices_per_series - 1) * slice_thickness / 2.0
                            + sl * slice_thickness, 3
                        )
                    )
                    pixels = gen(sl, slices_per_series, image_size, image_size)
                    write_dicom(
                        filepath        = filepath,
                        pixel_data      = pixels,
                        modality        = modality,
                        patient_id      = pid,
                        patient_name    = pname,
                        study_uid       = suid,
                        series_uid      = series_uid,
                        series_number   = series_num,
                        instance_number = sl + 1,
                        slice_location  = slice_loc,
                        slice_thickness = slice_thickness,
                        study_date      = sdate,
                        study_time      = study_time,
                    )
                    generation_status["progress"] += 1

        generation_status["message"] = "Generation complete!"
        generation_status["done"] = True
        generation_status["running"] = False
        generation_status["log"].append(
            f"\nDone! {total_series_count} series "
            f"({total_series_count * slices_per_series} DICOM files) "
            f"written to:\n  {output_dir}"
        )

    except Exception as exc:
        generation_status["error"] = str(exc)
        generation_status["running"] = False
        generation_status["done"] = True
        generation_status["log"].append(f"ERROR: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dcm_generator.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    if generation_status["running"]:
        return jsonify({"error": "Generation already in progress"}), 409

    data = request.get_json(force=True)

    # Validate
    required = ["modality", "num_studies", "num_series",
                 "patient_id_mode", "study_uid_mode", "date_mode"]
    for key in required:
        if key not in data:
            return jsonify({"error": f"Missing required field: {key}"}), 400

    if data["modality"] not in ("CT", "MR"):
        return jsonify({"error": "Modality must be CT or MR"}), 400

    try:
        ns = int(data["num_studies"])
        nser = int(data["num_series"])
        if ns < 1 or ns > 20:
            raise ValueError("num_studies must be 1-20")
        if nser < 1 or nser > 20:
            raise ValueError("num_series must be 1-20")
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    reset_status()
    generation_status["running"] = True

    t = threading.Thread(target=generate_dicoms_thread, args=(data,), daemon=True)
    t.start()

    return jsonify({"status": "started"})


@app.route("/api/status")
def api_status():
    return jsonify({
        "running":    generation_status["running"],
        "progress":   generation_status["progress"],
        "total":      generation_status["total"],
        "message":    generation_status["message"],
        "done":       generation_status["done"],
        "error":      generation_status["error"],
        "output_dir": generation_status["output_dir"],
        "log":        generation_status["log"],
    })


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("\n  DICOM Web Generator running at  http://127.0.0.1:5050\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
